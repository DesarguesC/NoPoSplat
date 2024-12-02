from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import AddLoss

@dataclass
class LossFlowCfg:
    weight: float

@dataclass
class LossFlowCfgWrapper:
    flow: LossFlowCfg

class LossFlow(AddLoss[LossFlowCfg, LossFlowCfgWrapper]):

    def __init__(self, cfg: LossFlowCfgWrapper):
        super().__init__(cfg)


    def forward(
        self,
        R: list,
        T: list,
    ) -> Float[Tensor, ""]:
        assert len(R) == len(T)
        I = torch.eyes((R[0].shape)).requires_grad_(False)
        L1_list = torch.tensor([
            torch.norm(R[i].T@R[i+1]-I, p='fro', dim=None, keepdim=True, out=None) + \
            (R[i].T@(T[i+1]-T[i]))**2 \
                for i in range(0,len(R)-1)
        ])
        return self.cfg.weight * L1_list.sum()
