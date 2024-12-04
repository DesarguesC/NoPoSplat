from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


class Loss(nn.Module, ABC, Generic[T_cfg, T_wrapper]):
    cfg: T_cfg
    name: str

    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__()

        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

    @abstractmethod
    def forward(
        self,
        prediction: DecoderOutput,  #
        batch: BatchedExample,      # g.t.
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        pass

class SmoothLoss(nn.Module, ABC, Generic[T_cfg, T_wrapper]):
    cfg: T_cfg
    name: str

    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__()

        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

    @abstractmethod
    def forward(
        self,
        Rotation_list: list,
        Translation_list: list,
    ) -> Float[Tensor, ""]:
        pass

class FlowLoss(nn.Module, ABC, Generic[T_cfg, T_wrapper]):
    cfg: T_cfg
    name: str

    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__()

        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

    @abstractmethod
    def forward(
        self,
        Flow_cam: Tensor,
        Flow_est: Tensor,
        Threshold: float,
    ) -> Float[Tensor, ""]:
        pass

def X2Flow_param(X):
    # 原文的意思是，输入带有各帧信息的X，计算FlowLoss
    # 这里的X是带有各个帧信息的list，调用返回
    pass
