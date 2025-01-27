import torch
import torch.nn as nn
import numpy as np
from einops import repeat, rearrange
from collections import OrderedDict
from torch.nn import init
import pdb
from timm.models.layers import DropPath, to_2tuple


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class Adapter(nn.Module):
    def __init__(self, channels=[640, 1280, 1280], nums_rb=3, frame=20, patches=320, ksize=1, sk=True, use_conv=False,
                 train_mode=True):
        # patches: p * n = (H//p_size) * (W//p_size) * 2 * n // 8
        super(Adapter, self).__init__()
        self.frame = frame
        self.shuffle = nn.PixelShuffle(2)  # amend ?
        self.unshuffle = nn.PixelUnshuffle(4)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    self.body.append(
                        ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(patches, channels[0], 3, 1, 1)
        if train_mode:
            self.initialize_weights()

    def forward(self, x):
        pdb.set_trace()
        x = rearrange(x, 'n b f p e -> b f (p n) e')  # 或者n和f放一起？
        x = self.unshuffle(x)
        # extract features
        features = []

        x = self.conv_in(x)  # # torch.Size([2, 320, 1280, 72])
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb \
                      + j
                x = self.body[idx](x)
            features.append(x)

        return features

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)), ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class StyleAdapter(nn.Module):

    def __init__(self, width=1024, context_dim=768, num_head=8, n_layes=3, num_token=4):
        super().__init__()

        scale = width ** -0.5
        self.transformer_layes = nn.Sequential(*[ResidualAttentionBlock(width, num_head) for _ in range(n_layes)])
        self.num_token = num_token
        self.style_embedding = nn.Parameter(torch.randn(1, num_token, width) * scale)
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, context_dim))

    def forward(self, x):
        # x shape [N, HW+1, C]
        style_embedding = self.style_embedding + torch.zeros(
            (x.shape[0], self.num_token, self.style_embedding.shape[-1]), device=x.device)
        x = torch.cat([x, style_embedding], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_layes(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, -self.num_token:, :])
        x = x @ self.proj

        return x


class ResnetBlock_light(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.block1 = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(in_c, in_c, 3, 1, 1)

    def forward(self, x):
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)

        return h + x


class extractor(nn.Module):
    def __init__(self, in_c, inter_c, out_c, nums_rb, down=False):
        super().__init__()
        self.in_conv = nn.Conv2d(in_c, inter_c, 1, 1, 0)
        self.body = []
        for _ in range(nums_rb):
            self.body.append(ResnetBlock_light(inter_c))
        self.body = nn.Sequential(*self.body)
        self.out_conv = nn.Conv2d(inter_c, out_c, 1, 1, 0)
        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=False)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        x = self.in_conv(x)
        x = self.body(x)
        x = self.out_conv(x)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=16, kernel_size=1, in_chans=6, embed_dim=576):
        # in_chans
        # 2 ray maps concatenated at channel
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
        pdb.set_trace()
        x = self.proj(x)
        return x


class Adapter_light(nn.Module):
    def __init__(self, channels=[640, 1280, 1280], nums_rb=3, frame=20, patches=1200, img_size=256, patch_size=4,
                 embed_dim=576):
        super(Adapter_light, self).__init__()
        self.frame = frame
        self.embedding_proj = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.shuffle = nn.PixelShuffle(4)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            if i == 0:
                # TODO: Here, the detail of the extractor
                self.body.append(
                    extractor(in_c=patches, inter_c=channels[i] // 4, out_c=channels[i], nums_rb=nums_rb, down=False))
            else:
                self.body.append(
                    extractor(in_c=channels[i - 1], inter_c=channels[i] // 4, out_c=channels[i], nums_rb=nums_rb,
                              down=True))
        self.body = nn.ModuleList(self.body)
        # self.conv_in = nn.Conv2d(patches, channels[0], 3, 1, 1)

    def forward(self, x):
        x = rearrange(x, 'n b f c h w -> b c (n f) h w')
        pdb.set_trace()
        x = self.embedding_proj(x)  # [b, c, f, h, w] -> [b, e, f, h1, w1]
        pdb.set_trace()
        x = rearrange(x, 'b e f h w -> b (e f) h w')  # 或者n和f放一起？
        x = self.shuffle(x)
        # unshuffle
        # x = rearrange(self.unshuffle(x), 'b f h w -> b p h w')
        # extract features
        features = []

        # x = self.conv_in(x)
        for i in range(len(self.channels)):
            x = self.body[i](x)
            features.append(x)

        return features


def main1():
    dec_feat = torch.randn((10, 2, 20, 256 ** 2 // 256 * 2, 576))  # [n, b, f, p, e] -> [b f (p n) e]
    dec_feat = dec_feat.to('cuda')
    pdb.set_trace()

    adapter_dec = Adapter(
        frame=20,
        patches=320,
        channels=[320, 160, 80],
        nums_rb=20,
        # ksize=1,
        # sk=True,
        # use_conv=False
    ).to('cuda')
    feature_ad = adapter_dec(dec_feat)
    pdb.set_trace()
    # Adapter: torch.Size([2, 640, 1280, 72]), torch.Size([2, 1280, 640, 36]), torch.Size([2, 1280, 320, 18])
    print(feature_ad[0].shape)


def main2():
    # 256 × 256
    pos = torch.ones((256, 256, 3)) * 3 / 255.
    dir = torch.ones((256, 256, 3)) * 3 / 20.

    pos = pos.to('cuda')
    dir = dir.to('cuda')
    pdb.set_trace()

    adapter_dec = Adapter_light(
        frame=20,
        patches=320,  # n * f * 3
        channels=[320, 160, 80],
        nums_rb=20,
        # ksize=1,
        # sk=True,
        # use_conv=False
    ).to('cuda')
    # [2 256, 256, 3]
    input = torch.cat([pos, dir], dim=-1)  # [256 256 3]
    input = input[None, :]
    input = repeat(input, '1 ... -> u ...', u=10 * 2 * 20)  # n * b * f
    input = rearrange(input, '(n b f) h w c -> n b f c h w', n=10, b=2).to('cuda')
    pdb.set_trace()
    # [n b f c H W]
    feature_ad = adapter_dec(input)

    pdb.set_trace()
    # Adapter_light: torch.Size([2, 640, 256, 256]), torch.Size([2, 1280, 128, 128]), torch.Size([2, 1280, 64, 64])
    print(feature_ad[0].shape)


# TODO: embeddings -> (H, W) ~ diffusion
class Adapter_Decoder(nn.Module):
    def __init__(self):
        ...


if __name__ == '__main__':
    main1()

