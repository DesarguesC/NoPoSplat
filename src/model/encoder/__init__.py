from typing import Optional
import pdb

from .encoder import Encoder
from .encoder_noposplat import EncoderNoPoSplatCfg, EncoderNoPoSplat
from .encoder_videosplat import EncoderVideoSplat
from .encoder_noposplat_multi import EncoderNoPoSplatMulti
from .visualization.encoder_visualizer import EncoderVisualizer
import torch
# from .encoder_vit import SpatialVivitModel, VideoMambaModel


ENCODERS = {
    "noposplat": (EncoderNoPoSplat, None),
    "noposplat_multi": (EncoderNoPoSplatMulti, None),
    "videosplat": (EncoderVideoSplat, None)
}



EncoderCfg = EncoderNoPoSplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    pdb.set_trace()
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
