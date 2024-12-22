# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional, Mapping, Any
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math, pdb

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import time, pdb
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from random import randint
import numpy as np
import cv2
from einops import repeat, rearrange
from basicsr.utils import tensor2img, img2tensor

MODEL_PATH = '../download/VisionMamba'
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
    "videomamba_b16_in1k": os.path.join(MODEL_PATH, "videomamba_b16_in1k_res224.pth")
}


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x
    

class VisionMamba(nn.Module):
    def __init__(
            self, 
            img_size=224, 
            patch_size=16, 
            depth=24, 
            embed_dim=192, 
            channels=3, 
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            # video
            kernel_size=1, 
            num_frames=8, 
            fc_drop_rate=0., 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
        ):
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # original init
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}
    
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None, **kwargs):
        x = self.patch_embed(x) # [b, embed_dim, frames, h/patch_size, w/patch_size]

        B, C, T, H, W = x.shape # check shapes
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C) # [(b * frames), (h/patch_size * w/patch_size), embed_dim]
        # [b * frames, 1, embed_dim]
        # TODO: 用CAT[cls_token, intrinsic_embed] ?
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        pdb.set_trace()
        x = x + self.pos_embed # [frames, num_patches+1, embed_dim]
        if 'intrinsic_embeddings' in kwargs:
            x = x + kwargs['intrinsic_embeddings'] # TODO: 可以加类似deformable gs里随迭代次数波动的正态噪声

        # temporal pos
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:] # num_patches+1 -> num_patches
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T) # [batch * num_patches, frames, embed_dim]
        x = x + self.temporal_pos_embedding # num_patches, frames, embed_dim
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T) # [batch, frames*num_patches, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1) # [batch, frames*num_patches+1, embed_dim]

        x = self.pos_drop(x)

        # mamba impl
        residual = None
        hidden_states = x
        pdb.set_trace()
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else: # √
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        pdb.set_trace() # hidden_states: [batch, frames*num_patches+1, embed_dim]
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else: # √
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        # hidden_states: [batch, frames*num_patches+1, embed_dim]
        pdb.set_trace()
        # return only cls token
        return hidden_states[:, 1:] if kwargs.get('destroy_head', False) else hidden_states[:, 0, :]

    def forward(self, x, inference_params=None, **kwargs):
        x = self.forward_features(x, inference_params, **kwargs)
        # [batch, embed_dim]
        pdb.set_trace()
        if not ('destroy_head' in kwargs and kwargs['destroy_head']):
            print('head used')
            x = self.head(self.head_drop(x), **kwargs) # destroy ?
        return x


def inflate_weight(weight_2d, time_dim, center=True):
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(model, state_dict, center=True):
    state_dict_3d = model.state_dict()
    pdb.set_trace()
    for k in state_dict.keys(): # 应该可以直接额外加，这里没有参数自动不填充新加的
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 3:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)
    
    del state_dict['head.weight']
    del state_dict['head.bias']
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)


@register_model
def videomamba_tiny(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, 
        embed_dim=192, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_t16_in1k"], map_location=device)
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def videomamba_small(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, 
        embed_dim=384, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_s16_in1k"], map_location=device)
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def videomamba_middle(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, 
        embed_dim=576, 
        depth=32, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_m16_in1k"], map_location=device)
        load_state_dict(model, state_dict, center=True)
    return model


"""
{
        'url': url: str = '',
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
}
"""

@register_model
def videomamba_base(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=768,
        depth=32,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_b16_in1k"], map_location=device)
        load_state_dict(model, state_dict, center=True)
    return model

mambas = {
    'tiny': videomamba_tiny,
    'small': videomamba_small,
    'middle': videomamba_middle,
    'base': videomamba_base
}

mamba_params = {
    'tiny':     {'patch_size': 16, 'embed_dim': 192, 'depth': 24},
    'small':    {'patch_size': 16, 'embed_dim': 384, 'depth': 24},
    'middle':   {'patch_size': 16, 'embed_dim': 576, 'depth': 32},
    'base':     {'patch_size': 16, 'embed_dim': 768, 'depth': 32},
}

def VideoMambaModel(mamba_choice='middle', num_frames=20, seed=4217, device='cuda'):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = mambas[mamba_choice](num_frames=num_frames).to(device)
    return model

class VideoMamba(nn.Module):
    def __init__(self, mamba_choice='middle', num_frames=20, seed=4217, device='cuda'):
        # TODO: 在制作数据的时候插入相同的stride
        # TODO: 如果在forward中随机丢弃stride？
        super().__init__()
        self.seed = seed
        self.device = device
        self.num_frames = num_frames
        self.mamba_model = VideoMambaModel(mamba_choice, num_frames, seed, device)
        # frame, batch, 3, 3
        self.embed_dim = mamba_params[mamba_choice]['embed_dim']
        self.intrinsic_encoder = nn.Sequential(
            nn.Linear(9 * num_frames, 2048),
            nn.Linear(2048, num_frames * self.embed_dim)
        )
        # TODO: maps to [b, embed_dim, frames, h / patch_size, w / patch_size]
        # 「See Line - 319'forward_features'」
        # forward: in_embed = self.intrinsic_encoder(context["intrinsics"].flatten(2))

    def forward(self,
                context: dict, # {'image':..., 'intrinsics':...}
                symmetrize_batch=False, # False
                return_views=False, # True
                ):
        b, f, _, h, w = context['video'].shaape
        b_, f_, _, h_, w_ = context['intrinsics'].shape
        assert h == w, f'width unequal to height: h = {h}, w = {w}'
        assert f == f_ and b == b_, (f'videos and intrinsics mismatched at the frame: (f, f_) = {(f, f_)}')
        pdb.set_trace()
        intrinsic_embed = self.intrinsic_encoder(rearrange(context['intrinsics'], 'b f h w -> b (f h w)'))
        intrinsic_embed = rearrange(intrinsic_embed.reshape((b_, f_, self.embed_dim)), 'b f e -> f b e')
        args_dict = {
            'intrinsic_embeddings': intrinsic_embed,
            'destroy_head': True
        }
        mamba_hidden_state = self.mamba_model(context['video'], **args_dict)
        # [batch, ]






        # TODO: dec1, dec2, shape1, shape2, view1, view2 = self.backbone(context, return_views=True)  # PE & Proj


    def load_intrinsics_encoder(self, weights_path: str) -> None:
        load_state_dict(self.intrinsic_encoder, weights_path)

    def save_state_dict(self, path, msg):
        filename = f'intrinsic_encoder_{msg}.pth'
        save_path = os.path.join(path, filename)
        try:
            state_dict = self.intrinsic_encoder.state_dict()
        except Exception as err:
            print(f'err: {err}')
            state_dict = self.intrinsic_encoder.module.state_dict() # multi gpus
        save_dict = {}
        for key, param in state_dict.items():
            if key.startswith('module.'):  # remove unnecessary 'module.'
                key = key[7:]
            save_dict[key] = param.cpu()
        torch.save(save_dict, save_path)
        # u can also choose to save optimizer.state_dict()




if __name__ == '__main__':


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 20
    img_size = 224 # 必须长宽相等
    folder_path = "../../../../datasets/point_odyssey/val/ani10_new_f/rgbs"

    # To evaluate GFLOPs, pleaset set `rms_norm=False` and `fused_add_norm=False`
    model = videomamba_middle(num_frames=num_frames).cuda()
    img_names = [os.path.join(folder_path, x) for x in os.listdir(folder_path)]
    img_list = [cv2.resize(cv2.imread(img), (img_size, img_size)) for img in img_names]
    batch_size = 1

    pdb.set_trace()
    monocular_tensor = [torch.cat([img2tensor(img_list[i])[None, :, :, :] for i in range(u, u + num_frames)]) for u in
                        range(len(img_list) - num_frames + 1)]
    u = randint(0, 100)
    video_tensor = torch.cat([monocular_tensor[i][None, :, :, :, :] for i in range(u, u + batch_size)], dim=0).to(device)
    # batch, frame, 3, H, W
    input = video_tensor.permute(0, 2, 1, 3, 4)

    # pdb.set_trace()
    # flops = FlopCountAnalysis(model, input)

    intrinsics = torch.randn((batch_size, num_frames, 3, 3)).to(device)
    encoder_fn = nn.Sequential(
            nn.Linear(9 * num_frames, 2048),
            nn.Linear(2048, num_frames * 576)
    ).to(device)

    intrinsic_embed = encoder_fn(rearrange(intrinsics, 'b f h w -> b (f h w)'))
    intrinsic_embed = rearrange(intrinsic_embed.reshape((batch_size, num_frames, 576)), 'b f e -> f b e')

    # test for splat inference
    k_dict = {
        'intrinsic_embeddings': intrinsic_embed,
        'destroy_head': True,
    }

    s = time.time()
    output = model(input, **k_dict)
    # [batch, embed_dim]

    # print(flop_count_table(flops, max_depth=1))
    print(f'Time Cost: {time.time() - s}')