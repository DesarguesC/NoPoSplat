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
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from basicsr.utils import tensor2img, img2tensor
from einops import rearrange, repeat
from transformers import VivitModel
from transformers.models.vivit.modeling_vivit import VivitEmbeddings, VivitPreTrainedModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator

class SpatialVivitEmbeddings(VivitEmbeddings):

    def forward(self, pixel_values, interpolate_pos_encoding: bool=False): # only 'pirnt' added
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        print(f'overwrite worked: embeddings.shape = {embeddings.shape}')
        cls_tokens = self.cls_token.tile([batch_size, 1, 1])
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # pdb.set_trace()
        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings

@add_start_docstrings(
    "The bare ViViT Transformer model outputting raw hidden-states without any specific head on top.",
)
class SpatialVivitModel(VivitModel):
    """
    for overwriting
    """
    def __init__(self, config, add_pooling_layer=True, new_embedding=SpatialVivitEmbeddings):
        super().__init__(config, add_pooling_layer)
        # pdb.set_trace()
        if new_embedding is not None:
            self.embeddings = new_embedding(config)

inf = float('inf')
from .encoder_noposplat import EncoderNoPoSplatCfg
class EncoderVivitSplat(Encoder[EncoderNoPoSplatCfg]):

    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderNoPoSplatCfg) -> None:
        super().__init__(cfg)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        backbone = SpatialVivitModel.from_pretrained("../google/vivit-b-16x2-kinetics400", attn_implementation="sdpa", torch_dtype=torch.float32)
        self.backbone = backbone.to(device)
        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

        self.gs_params_head_type = cfg.gs_params_head_type

        self.set_center_head(output_mode='pts3d-dynamic', head_type='dpt', landscape_only=True,
                           depth_mode=('exp', -inf, inf), conf_mode=None,)
        # -> self.downstream_head1, self.downstream_head2, self.head1, self.head2
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)
        # -> self.gaussian_param_head, self.gaussian_param_head2










