from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import FlowLoss

@dataclass
class LossFlowCfg:
    weight: float

@dataclass
class LossFlowCfgWrapper:
    flow: LossFlowCfg

class LossFlow(FlowLoss[LossFlowCfg, LossFlowCfgWrapper]):

    def __init__(self, cfg: LossFlowCfgWrapper):
        super().__init__(cfg)


    def forward(
        self,
        Flow_cam: Tensor,
        Flow_est: Tensor,
        Threshold: float
    ) -> Float[Tensor, ""]:
    # 断续地包含training strategy中的t←t'，求和在外部实现（这里只实现某一项的FlowLoss）
        delta = Flow_cam - Flow_est
        S = delta[delta < Threshold] # True / False
        return S * (delta)
