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


def get_backbone(cfg: BackboneCfg, d_in: int = 3) -> nn.Module:
    backbone = BACKBONES[cfg.name]
    keys = inspect.signature(backbone.__init__).parameters
    pdb.set_trace()
    cfg_valid = {k: cfg[k] for k in keys}
    if isinstance(backbone, Optional[BackboneResnet, BackboneDino]):
        cfg_valid['d_in'] = d_in

    # TODO: write a cfg template | use variables in 'opt'
    return backbone(**cfg_valid) if \
        isinstance(backbone, Optional[BackboneResnet, BackboneDino, AsymmetricCroCo, AsymmetricCroCoMulti, VideoMamba]) \
            else None
