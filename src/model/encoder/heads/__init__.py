# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# head factory
# --------------------------------------------------------
from .dpt_gs_head import create_gs_dpt_head
from .linear_head import LinearPts3d
from .dpt_head import create_dpt_head
import pdb

def head_factory(head_type, output_mode, net, has_conf=False, out_nchan=3, static_required=False):
    """" build a prediction head for the decoder
    """
    if head_type == 'linear' and output_mode == 'pts3d':
        return LinearPts3d(net, has_conf)
    elif head_type == 'dpt' and output_mode == 'pts3d': # √
        return create_dpt_head(net, has_conf=has_conf)
    # TODO: static & dynamic 的具体实现有待商榷，以及时间编码如何加入；先结局前面的代码结构问题
    elif head_type == 'dpt-video' and output_mode == 'pts3d-video': # only sole dynamic head
        assert not static_required, 'illegal access to static gaussian head.'
        pdb.set_trace()
        # net: VideoMamba
        return create_dpt_head(net, has_conf=has_conf, use_mamba=True, static_required=static_required) # only sole dynamic head
    elif head_type == 'dpt-video' and output_mode == 'pts3d-video-hierarchical': # 1 dynamic head + 1 static head
        if static_required:
            ... # only sole static head
        else:
            pdb.set_trace()
            # net: VideoMamba
            return create_dpt_head(net, has_conf=has_conf, use_mamba=True, static_required=static_required) # only sole dynamic head
    elif head_type == 'dpt' and output_mode == 'gs_params':
        return create_dpt_head(net, has_conf=False, out_nchan=out_nchan, postprocess_func=None)
    elif head_type == 'dpt_gs' and output_mode == 'gs_params':
        return create_gs_dpt_head(net, has_conf=False, out_nchan=out_nchan, postprocess_func=None)
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")
