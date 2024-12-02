from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import SmoothLoss


@dataclass
class LossSmoothCfg:
    weight: float

@dataclass
class LossSmoothCfgWrapper:
    smooth: LossSmoothCfg


class LossFlow(SmoothLoss[LossSmoothCfg, LossSmoothCfgWrapper]):

    def __init__(self, cfg: LossSmoothCfgWrapper):
        super().__init__(cfg)


    def forward(
        self,
        R: list,
        T: list,
    ) -> Float[Tensor, ""]:
    # 连续地包含所有的t，求和在此处实现
        assert len(R) == len(T)
        I = torch.eyes((R[0].shape)).requires_grad_(False)
        L1_list = torch.tensor([
            torch.norm(R[i].T@R[i+1]-I, p='fro', dim=None, keepdim=True, out=None) + \
            (R[i].T@(T[i+1]-T[i]))**2 \
                for i in range(0,len(R)-1)
        ])
        return self.cfg.weight * L1_list.sum()
