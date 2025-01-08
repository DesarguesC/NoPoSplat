from enum import Enum, unique
import numpy as np
import cv2
import torch, pdb
from basicsr.utils import img2tensor
from ...util import resize_numpy_image, get_resize_shape, Inter
from PIL import Image
from torch import autocast
from ...util import Inter



@unique
class ExtraCondition(Enum):
    ray = 0
    feature = 1

# Useless
def get_cond_model(opt, cond_type: ExtraCondition):
    if cond_type == ExtraCondition.sketch:
        from ...modules.extra_condition.model_edge import pidinet
        model = pidinet()
        ckp = torch.load('models/table5_pidinet.pth', map_location='cpu')['state_dict']
        model.load_state_dict({k.replace('module.', ''): v for k, v in ckp.items()}, strict=True)
        model.to(opt.device)
        return model
    elif cond_type == ExtraCondition.seg:
        raise NotImplementedError
    elif cond_type == ExtraCondition.keypose:
        import mmcv
        from mmdet.apis import init_detector
        from mmpose.apis import init_pose_model
        det_config = 'configs/mm/faster_rcnn_r50_fpn_coco.py'
        det_checkpoint = 'models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        pose_config = 'configs/mm/hrnet_w48_coco_256x192.py'
        pose_checkpoint = 'models/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
        det_config_mmcv = mmcv.Config.fromfile(det_config)
        det_model = init_detector(det_config_mmcv, det_checkpoint, device=opt.device)
        pose_config_mmcv = mmcv.Config.fromfile(pose_config)
        pose_model = init_pose_model(pose_config_mmcv, pose_checkpoint, device=opt.device)
        return {'pose_model': pose_model, 'det_model': det_model}
    elif cond_type == ExtraCondition.depth:
        from ...modules.extra_condition.midas.api import MiDaSInference
        model = MiDaSInference(model_type='dpt_hybrid').to(opt.device)
        return model
    elif cond_type == ExtraCondition.canny:
        return None
    elif cond_type == ExtraCondition.style:
        from transformers import CLIPProcessor, CLIPVisionModel
        pdb.set_trace()
        version = 'openai/clip-vit-large-patch14' # deal with download paths
        processor = CLIPProcessor.from_pretrained(version)
        clip_vision_model = CLIPVisionModel.from_pretrained(version).to(opt.device)
        return {'processor': processor, 'clip_vision_model': clip_vision_model}
    elif cond_type == ExtraCondition.color:
        return None
    elif cond_type == ExtraCondition.openpose:
        from ...modules.extra_condition.openpose.api import OpenposeInference
        model = OpenposeInference().to(opt.device)
        return model
    else:
        raise NotImplementedError


def get_adapter_feature(inputs, adapters):
    # input: condition
    ret_feat_map = None
    ret_feat_seq = None
    if not isinstance(inputs, list):
        inputs = [inputs]
        adapters = [adapters]

    for input, adapter in zip(inputs, adapters):
        cur_feature = adapter['model'](input)
        if isinstance(cur_feature, list):
            if ret_feat_map is None:
                ret_feat_map = list(map(lambda x: x * adapter['cond_weight'], cur_feature))
            else:
                ret_feat_map = list(map(lambda x, y: x + y * adapter['cond_weight'], ret_feat_map, cur_feature))
        else:
            if ret_feat_seq is None:
                ret_feat_seq = cur_feature
            else:
                ret_feat_seq = torch.cat([ret_feat_seq, cur_feature], dim=1)

    return ret_feat_map, ret_feat_seq
