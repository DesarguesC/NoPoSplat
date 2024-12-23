from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch, pdb
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .encoder_noposplat import EncoderNoPoSplatCfg
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg


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

    def __init__(self, cfg: EncoderNoPoSplatCfg) -> None:
        super().__init__(cfg)
        self.head_control_type = UNDEFINED_VALUE # unset
        """
            1: 只有一种head，需要加上时间编码接口
            2: 有两种head，静态head无需时间编码，动态head需要时间编码接口（与1用的相同）
        """

        self.backbone = get_backbone(cfg.backbone, 3) # VideoMamba

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.enc_patch_size
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

        self.gs_params_head_type = cfg.gs_params_head_type

        head_type = 'dpt-video' if cfg.gs_params_head_type == 'dpt-video' else 'dpt'
        output_type = 'pts3d-video' if cfg.gs_params_head_type == 'dpt-video' else \
                      'pts3d-video-hierarchical' if cfg.gs_params_head_type == 'dpt-video-hierarchical' else 'dpt'

        self.set_center_head(output_mode=output_type, head_type=head_type, landscape_only=True, depth_mode=('exp', -inf, inf), conf_mode=None, )
        # -> self.downstream_head1, self.downstream_head2, self.head1, self.head2
        self.set_gs_params_head(cfg, head_type, output_type)
        # -> self.gaussian_param_head, self.gaussian_param_head2
    
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

    # def _downstream_head(self, head_name: str, decout, img_shape, ray_embedding=None):
    #     # decout: decoded token list
    #     # B, S, D = decout[-1].shape
    #     # img_shape = tuple(map(int, img_shape))
    #     head = getattr(self, f'head_{head_name}')
    #     return head(decout, img_shape, ray_embedding=ray_embedding)
    
    # SOLE_HEAD works at ↓
    def forward_one(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:

        """
            Available:
                center-head:
                    self.downstream_head ~ self.head_sole
                param-head:
                    self.gaussian_param_head_sole
        """

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

    # MULTI_HEAD works at ↓
    def forward_two(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:

        """
            Available:
                center-head:
                    self.downstream_head_static ~ self.head_static
                    self.downstream_head_dynamic ~ self.head_dynamic
                param-head:
                    self.gaussian_param_head_static
                    self.gaussian_param_head_dynamic
        """
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        # TODO: [AIR] 要有3个head，多一个diffusion的
        #
        dec_feature_list, views = self.backbone(context, return_views=True) # PE & Proj
        # src.model/encoder/backbone/backbone_croco.py - line: 222
        # 完成了：嵌入、feature回归、解码 -> tokens
        """
            views = {
                'video': context['video'],
                'position': pos,
                'embeddings': embeddings
            }
        """

        pdb.set_trace()
        video = views['video']
        shpe = views['shape']
        with torch.cuda.amp.autocast(enabled=False):
            all_mean_res = []
            all_other_params = []
            res1 = self._downstream_head(1, [tok[:, 0].float() for tok in dec_feat], shape[:, 0])
            all_mean_res.append(res1)
            for i in range(1, v):
                res2 = self._downstream_head(2, [tok[:, i].float() for tok in dec_feat], shape[:, i])
                all_mean_res.append(res2)

            # for the 3DGS heads
            if self.gs_params_head_type == 'dpt_gs':
                GS_res1 = self.gaussian_param_head([tok[:, 0].float() for tok in dec_feat],
                                                   all_mean_res[0]['pts3d'].permute(0, 3, 1, 2), images[:, 0, :3],
                                                   shape[0, 0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                all_other_params.append(GS_res1)
                for i in range(1, v):
                    GS_res2 = self.gaussian_param_head2([tok[:, i].float() for tok in dec_feat],
                                                        all_mean_res[i]['pts3d'].permute(0, 3, 1, 2), images[:, i, :3],
                                                        shape[0, i].cpu().tolist())
                    GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
                    all_other_params.append(GS_res2)
            else:
                raise NotImplementedError(f"unexpected {self.gs_params_head_type=}")

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
            xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)

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
    ) -> Gaussians:
        if self.head_control_type == SOLE_HEAD:
            return self.forward_one(context, global_step, visualization_dump)
        elif self.head_control_type == MULTI_HEAD:
            return self.forward_two(context, global_step, visualization_dump)
        else:
            raise NotImplementedError
