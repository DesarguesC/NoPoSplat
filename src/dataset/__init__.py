from dataclasses import fields
import pdb
from torch.utils.data import Dataset

from .dataset_scannet_pose import DatasetScannetPose, DatasetScannetPoseCfgWrapper
from .dataset_point_odyssey import DatasetPointOdyssey, DatasetPointOdysseyCfgWrapper
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg, DatasetRE10kCfgWrapper, DatasetDL3DVCfgWrapper, \
    DatasetScannetppCfgWrapper

from ..misc.step_tracker import StepTracker
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str: Dataset] = {
    "re10k": DatasetRE10k,
    "dl3dv": DatasetRE10k,
    "scannetpp": DatasetRE10k,
    "scannet_pose": DatasetScannetPose,
    'point_odyssey': DatasetPointOdyssey
}


DatasetCfgWrapper = DatasetRE10kCfgWrapper | DatasetDL3DVCfgWrapper | DatasetScannetppCfgWrapper | DatasetScannetPoseCfgWrapper | DatasetPointOdysseyCfgWrapper
DatasetCfg = DatasetRE10kCfg


def get_dataset(
    cfgs: list[DatasetCfgWrapper],
    stage: Stage,
    step_tracker: StepTracker | None,
) -> list[Dataset]:
    datasets = []
    for cfg in cfgs:
        (field,) = fields(type(cfg))
        cfg = getattr(cfg, field.name)

        view_sampler = get_view_sampler(
            cfg.view_sampler,
            stage,
            cfg.overfit_to_scene is not None,
            cfg.cameras_are_circular,
            step_tracker,
        )
        dataset = DATASETS[cfg.name](cfg, stage, view_sampler)
        datasets.append(dataset)

    return datasets
