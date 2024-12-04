
from typing import Optional

from .loss import Loss, SmoothLoss, FlowLoss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_smooth import LossSmooth, LossSmoothCfgWrapper
from .loss_flow import LossFlow, LossFlowCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossSmoothCfgWrapper: LossSmooth,
    LossFlowCfgWrapper: LossFlow
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossSmoothCfgWrapper | LossFlowCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss | FlowLoss | SmoothLoss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
