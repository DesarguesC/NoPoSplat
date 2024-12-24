from typing import Any, Optional
import torch.nn as nn

from .backbone import Backbone
from .backbone_croco_multiview import AsymmetricCroCoMulti
from .backbone_dino import BackboneDino, BackboneDinoCfg
from .backbone_resnet import BackboneResnet, BackboneResnetCfg
from .backbone_croco import AsymmetricCroCo, BackboneCrocoCfg
from .backbone_videomamba import VideoMamba

BACKBONES: dict[str, Backbone[Any]] = {
    "resnet": BackboneResnet,
    "dino": BackboneDino,
    "croco": AsymmetricCroCo,
    "croco_multi": AsymmetricCroCoMulti,
    'videomamba': VideoMamba
}

BackboneCfg = BackboneResnetCfg | BackboneDinoCfg | BackboneCrocoCfg


def get_backbone(cfg: BackboneCfg, d_in: int = 3) -> nn.Module:
    backbone = BACKBONES[cfg.name]
    return backbone(cfg, d_in) if isinstance(backbone, Optional[BackboneResnet, BackboneDino]) \
            else backbone(cfg) if isinstance(backbone, Optional[AsymmetricCroCo, AsymmetricCroCoMulti]) \
            else backbone() if isinstance(backbone, VideoMamba)\
            else None
