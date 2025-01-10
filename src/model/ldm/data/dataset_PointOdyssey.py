import json
import cv2
import os, pdb
from copy import copy
import torch.nn as nn
from basicsr.utils import img2tensor, tensor2img
from typing import Optional
from random import randint
import numpy as np
from einops import repeat, rearrange

"""
<device>
context = {
    'video': [b f c h w],
    'intrinsics': [b f 3 3],
    'ray': [...], # create camera ray map
    '':
}

intrinsics -> 
        'K_dict': {
                'intrinsic_embeddings': [f b e],
                'destroy_head': True,
                'return_pos': True,
        }
        automatically done at VideoMamba backbone

"""


class PointOdysseyDataset():
    def __init__(self, folder_root, type: str = 'train', frame: int = 8):
        super(PointOdysseyDataset, self).__init__()
        assert type in ['train', 'test', 'val']
        self.frame = frame
        folder_root = os.path.join(folder_root, type)
        folder_list = os.listdir(folder_root)
        folder_list = [u for u in folder_list if (os.path.isdir(u) and '.' not in u)]
        self.data_dict = {}
        self.name_list = []
        self.data_list_dict = {
            'rgb': [],
            'depth': [],
            'normal': [],
            'mask': [],
        }
        self.len_prefix = [0]
        for k in folder_list:
            rgb_list = [u for u in os.listdir(os.path.join(folder_root, k, 'rgbs')) if 'jpg' in u]
            depth_list = [u for u in os.listdir(os.path.join(folder_root, k, 'depths')) if 'jpg' in u]
            normal_list = [u for u in os.listdir(os.path.join(folder_root, k, 'normals')) if 'jpg' in u]
            mask_list = [u for u in os.listdir(os.path.join(folder_root, k, 'masks')) if 'jpg' in u]
            self.data_dict[k] = {
                'video': os.path.join(folder_root, f'{k}.mp4'),
                'annotation': (os.path.join(folder_root, k, 'anno.npz'), os.path.join(folder_root, k, 'info.npz')),
            }

            rgb_list.sort()
            depth_list.sort()
            normal_list.sort()
            mask_list.sort()

            self.data_list_dict['rgb'].append([rgb_list[idx:idx+frame] for idx in range(len(rgb_list)-frame)])
            self.data_list_dict['depth'].append([depth_list[idx:idx+frame] for idx in range(len(depth_list)-frame)])
            self.data_list_dict['normal'].append([normal_list[idx:idx+frame] for idx in range(len(normal_list)-frame)])
            self.data_list_dict['mask'].append([mask_list[idx:idx+frame] for idx in range(len(mask_list)-frame)])

            self.len_prefix.append(len(self.data_list_dict['rgb']))

            self.name_list.append(k)



    def get_anno(self, idx):
        for i in range(len(self.len_prefix)-1):
            if idx >= self.len_prefix[i] and idx < self.len_prefix[i+1]:
                return self.data_dict[self.name_list[i]]['annotation']

        raise RuntimeError('index out of range of PointOdyssey Dataset')


    def __getitem__(self, idx):
        ...


