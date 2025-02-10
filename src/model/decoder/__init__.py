from .decoder import Decoder
from dataclasses import dataclass
from typing import Literal

@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool

try:
    from .decoder_splatting_cuda import DecoderSplattingCUDA
    DECODERS = {
        "splatting_cuda": DecoderSplattingCUDA,
    }

except Exception as err:
    print(err)


DecoderCfg = DecoderSplattingCUDACfg
def get_decoder(decoder_cfg: DecoderCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg)
