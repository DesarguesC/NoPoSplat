import os.path
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional
from torch import autocast

import torch, pdb
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory

from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .encoder_noposplat import EncoderNoPoSplatCfg
from ..ldm import *



inf = float('inf')
UNDEFINED_VALUE = 'yet this value has not been set'
SOLE_HEAD = 'now we are using single dynamic gaussian head'
MULTI_HEAD = 'now we are using multi gaussian heads: static & dynamic'


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat

class EncoderVideoSplat(Encoder[EncoderNoPoSplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderNoPoSplatCfg, args = None) -> None:
        super().__init__(cfg)
        self.head_control_type = UNDEFINED_VALUE # unset
        """
            1: 只有一种head，需要加上时间编码接口
            2: 有两种head，静态head无需时间编码，动态head需要时间编码接口（与1用的相同）
        """
        # pdb.set_trace()
        self.backbone = get_backbone(cfg.backbone, 3, args) # VideoMamba
        self.train_mode = args.train_mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # inference
        self.frame = args.frame
        self.cond_weight = args.cond_weight
        self.batch = getattr(args, 'batch_size', 1)

        self.pose_free = cfg.pose_free

        # if self.pose_free:
        #     self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        # else:
        #     self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.enc_patch_size
        # self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

        # self.gs_params_head_type = cfg.gs_params_head_type

        self.sd_opt = make_options(train_mode=args.train_mode)
        self.sd_model, self.sampler, self.sd_cfg = get_sd_models(self.sd_opt, return_cfg=True, debug=False)
        # with stable-diffusion loaded
        # pdb.set_trace()
        self.adapter_dict = get_latent_adapter(self.sd_opt, train_mode=args.train_mode, cond_type=self.sd_opt.allow_cond, device=self.device)
        # 'model': list 'cond_weight': list
        # pdb.set_trace()
        if not args.train_mode:
            assert os.path.exists(args.pretrained_weights)
            weights = torch.load(args.pretrained_weights)
            ad_0_ckpt, ad_1_ckpt = self.backbone.load_encoder_and_decoder(weights)
            # pdb.set_trace()
            self.adapter_dict['model'][0].load_state_dict(ad_0_ckpt, strict=False)
            self.adapter_dict['model'][1].load_state_dict(ad_1_ckpt, strict=False)

            del weights


        # TODO: params = list(xx.backbone.parameters()) + list(xx.adapter_dict['ray']) + list(xx.adapter_dict['feature'])
        # optimizer = torch.optim.AdamW(params, lr=...)
        """
        head_type = 'dpt-video' if cfg.gs_params_head_type == 'dpt-video' else 'dpt'
        output_type = 'pts3d-video' if cfg.gs_params_head_type == 'dpt-video' else \
                      'pts3d-video-hierarchical' if cfg.gs_params_head_type == 'dpt-video-hierarchical' else 'dpt'

        self.set_center_head(output_mode=output_type, head_type=head_type, landscape_only=True, depth_mode=('exp', -inf, inf), conf_mode=None, )
        # -> self.downstream_head1, self.downstream_head2, self.head1, self.head2
        self.set_gs_params_head(cfg, head_type, output_type)
        # -> self.gaussian_param_head, self.gaussian_param_head2
        """
    
    def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        # allocate heads
        # TODO: 先考虑只用static head，dynamic用来算[R|T]
        if output_mode == 'pts3d-video' and head_type == 'dpt-video':
            self.head_control_type = SOLE_HEAD

            self.downstream_head = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode), static_required=False) # 返回sole dynamic head
            self.head_sole = transpose_to_landscape(self.downstream_head, activate=landscape_only)

        elif output_mode == 'pts3d-video-hierarchical' and head_type == 'dpt-video':
            self.head_control_type = MULTI_HEAD

            self.downstream_head_static = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode), static_required=True) # 返回static head
            self.head_static = transpose_to_landscape(self.downstream_head_static, activate=landscape_only)

            self.downstream_head_dynamic = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode), static_required=False) # 返回dynamic head
            self.head_dynamic = transpose_to_landscape(self.downstream_head_dynamic, activate=landscape_only)

        else:
            raise NotImplementedError('check parameters at configs, listed in \'param.txt\'')
    def set_gs_params_head(self, cfg, head_type, output_mode):
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )
            self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)
        elif head_type == 'dpt':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)  # for view1 3DGS
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)  # for view2 3DGS
        elif head_type == 'dpt_gs':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
        elif self.head_control_type == SOLE_HEAD:
            self.gaussian_param_head_sole = head_factory(head_type, output_mode, self.backbone, has_conf=False, static_required=False) # 返回sole dynamic head
        elif self.head_control_type == MULTI_HEAD:
            self.gaussian_param_head_static = head_factory(head_type, output_mode, self.backbone, has_conf=False, static_required=True) # 返回static head
            self.gaussian_param_head_dynamic = head_factory(head_type, output_mode, self.backbone, has_conf=False, static_required=False) # 返回dynamic head
        else:
            raise NotImplementedError(f"unexpected {head_type=}")
    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    # only InstanceMaskedAttention_Head works at ↓
    def forward_one(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ):

        """
            Available:
                center-head:
                    self.downstream_head ~ self.head_sole
                param-head:
                    self.gaussian_param_head_sole
        """

        with torch.cuda.amp.autocast(enabled=False):
            # 原来noposplat的cross-attention思路是用第0帧查询后续所有帧
            # 在adapter forward中做'n b f p e -> b f (p n) e'
            # feature_dec = self.adapter_list[0](ray_maps['pos'], dec_feature)
            # feature_ray = self.adapter_list[1](ray_maps['dir'], dec_feature) # 也可以直接把dir做encode之后拼到sd的backbone上
            #
            # adapter_feature = [feature_dec, feature_ray] # List[list]
            # L = len(adapter_feature[0])
            # concat_feat = [
            #     cond_weight[0] * adapter_feature[0][i] + cond_weight[1] * adapter_feature[1][i]
            #         for i in range(L)
            # ]
            # # b = 1 when inference, b <- b * f
            # concat_feat = [u.squeeze(0) for u in concat_feat]

            dec_feat, _ = self.backbone(context=context, return_views=True)
            mamba_feat = self.adapter_dict['model'][0](dec_feat)
            with torch.inference_mode(), \
                    self.sd_model.ema_scope(), \
                    autocast('cuda'):
                ray_feat = self.adapter_dict['model'][1](rearrange(context['ray'], '(f u) h w -> f u h w', f=self.frame)) # '(n c f) h w -> f (n c) h w'

                # Add shape validation
                assert len(mamba_feat) == len(ray_feat), "Feature list length mismatch"
                features_adapter = [
                    self.cond_weight[0] * mamba_feat[i] + self.cond_weight[1] * ray_feat[i]
                    for i in range(len(mamba_feat))
                ]
                # self.batch * self.frame
                x_samples = diffusion_inference(self.sd_opt, self.sd_model, self.sampler, features_adapter, batch_size=1) # [b f c h w]
        # pdb.set_trace()
        return x_samples

    # 我倾向于这个会work
    # Gaussian_Head + VehiclePose_Head + InstanceMaskedAttention_Head works at ↓
    # InstanceMaskedAttention_Head work了以后再看能否用continuous attention优化
    def forward_two(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:
        video = context['video']
        device = video.device
        b, f, _, h, w = video.shape

        # TODO: [AIR] 要有3个head，多一个diffusion的
        dec_feature, views = self.backbone(context, return_views=True) # PE & Proj
        """
            views = {
                'shape': shape # [batch, frame, channel(=3)]
                'video': context['video'],  # [batch, frames, height, width]
                'position': pos,            # [batch, frames, num_patcher, embed_dim]
                'embeddings': embeddings    # [num_patches, frames, batch, embed_dim]
            }
        """

        video = views['video']
        shape = views['shape']
        pdb.set_trace()
        with torch.cuda.amp.autocast(enabled=False):
            # only sole dynamic head
            all_mean_res = [
                self.head_sole(
                    [tok[:, i].float() for tok in dec_feature],
                    shape[:, i], ray_embedding=None
                )
                for i in range(f)
            ]
            all_other_params = [
                rearrange(self.gaussian_param_head_sole(
                    [tok[:, i].float() for tok in dec_feature],
                    all_mean_res[i]['pts3d'].permute(0, 3, 1, 2),
                    video[:, i, :3],
                    shape[0, i].cpu().tolist()
                ), "b d h w -> b (h w) d") # TODO: check the shape ← check & modify DPTAdapter
                for i in range(f)
            ]
            # TODO: 形状都需要检查 → 'pts3d' ???

        pts_all = [all_mean_res_i['pts3d'] for all_mean_res_i in all_mean_res]
        pts_all = torch.stack(pts_all, dim=1)
        pts_all = rearrange(pts_all, "b v h w xyz -> b v (h w) xyz")
        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces

        depths = pts_all[..., -1].unsqueeze(-1)

        gaussians = torch.stack(all_other_params, dim=1)
        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

        # Convert the features and depths into Gaussians.
        if self.pose_free:
            gaussians = self.gaussian_adapter.forward(
                pts_all.unsqueeze(-2),
                depths,
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
            )
        else:
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            xy_ray = xy_ray[None, None, ...].expand(b, f, -1, -1, -1)

            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
                (h, w),
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            visualization_dump["means"] = rearrange(
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            )
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )



    def forward(
            self,
            context: dict,
            global_step: int = 0,
            visualization_dump: Optional[dict] = None,
            parrallel: bool = False
    ):
        if self.head_control_type == SOLE_HEAD:
            return self.forward_one(context, global_step, visualization_dump)
        elif self.head_control_type == MULTI_HEAD:
            return self.forward_two(context, global_step, visualization_dump)
        elif self.head_control_type == UNDEFINED_VALUE: # √
            return self.forward_one(context, global_step, visualization_dump)
        else:
            raise NotImplementedError
