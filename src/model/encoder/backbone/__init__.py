from typing import Any, Optional
import torch.nn as nn
import inspect, pdb
from .backbone import Backbone
from .backbone_croco_multiview import AsymmetricCroCoMulti
from .backbone_dino import BackboneDino, BackboneDinoCfg
from .backbone_resnet import BackboneResnet, BackboneResnetCfg
from .backbone_croco import AsymmetricCroCo, BackboneCrocoCfg, BackboneMambaCfg
from .backbone_videomamba import VideoMamba

BACKBONES: dict[str, Backbone[Any]] = {
    "resnet": BackboneResnet,
    "dino": BackboneDino,
    "croco": AsymmetricCroCo,
    "croco_multi": AsymmetricCroCoMulti,
    'videomamba': VideoMamba
}

BackboneCfg = BackboneResnetCfg | BackboneDinoCfg | BackboneCrocoCfg | BackboneMambaCfg


def get_backbone(cfg: BackboneCfg, d_in: int = 3, args = None) -> nn.Module:
    # backbone = BACKBONES[cfg.name]
    # keys = inspect.signature(backbone.__init__).parameters
    # pdb.set_trace()
    # cfg_valid = {k: cfg[k] for k in keys} # check cfg.name
    # if isinstance(backbone, Optional[BackboneResnet, BackboneDino]):
    #     cfg_valid['d_in'] = d_in
    # if isinstance(backbone, VideoMamba):
    return VideoMamba(
            mamba_choice='base',
            num_frames=args.frame,
            seed=args.seed,
            dec_embed_dim=args.dec_embed_dim,
            dec_depth=args.dec_depth,
            dec_num_heads=args.dec_num_heads,
            mlp_ratio=args.mlp_ratio,
            norm_im2_in_dec=args.norm_im2_in_dec,
            pos_embed=args.pos_embed,
            decoder_weights_path=args.decoder_weights_path,
            device=args.device,
        )

    # TODO: write a cfg template | use variables in 'opt'
    return backbone(cfg, d_in) if \
        isinstance(backbone, Optional[BackboneResnet, BackboneDino]) \
        else backbone(cfg) if isinstance(backbone, Optional[AsymmetricCroCo, AsymmetricCroCoMulti]) \
        else None
