import json
import cv2
import os, pdb, torch
import open3d as o3d
from copy import copy
import torch.nn as nn
from basicsr.utils import img2tensor, tensor2img
from functools import partial
from torch.nn import functional as F
from typing import Optional
from random import randint
import numpy as np
from .utils import positional_encode, camera2ray
from tqdm import tqdm


K_veh = np.array([[2730.3754, 0., 960.], [0., 2730.3754, 600.], [0., 0., 1.]]) # -> create camera ray
# Basler ace acA1920-50gc, 1920×1200, Sony IMX174 with 16 mm lenses
K_inf = np.array([[1365.1877, 0., 960.], [0., 1365.1877, 600.], [0., 0., 1.]]) # video mamba input
# Basler ace acA1920-50gc, 1920×1200, Sony IMX174 with 8 mm lenses


"""
video list: 
    [
        {
            'infrastructure-side': [<img path>, ...],
            'vehicle-side': [<img path>, ...],
            'R': [3 3],
            'T': [3 1],
        },
        ...
        
    ]


"""
def create_dict(K):
    return {
        'height': int(K[0,-1]*2),
        'width': int(K[1,-1]*2),
        'fx': K[0,0], 'fy': K[1,1],
        'cx': K[0,-1], 'cy': K[1,-1]
    }

def render_pcd_view_img(pcd_point, K, trans, shape):

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=shape[1], height=shape[0])
    vis.add_geometry(pcd_point)
    pdb.set_trace()
    vis.get_render_option().point_size = 100.
    camera = o3d.camera.PinholeCameraParameters()
    camera.extrinsic = trans
    camera.intrinsic.set_intrinsics(**create_dict(K))

    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(camera)
    vis.poll_events()
    vis.update_renderer()

    return np.asarray(vis.capture_screen_float_buffer(do_render=True))


def capture_point_cloud_render(pcd, K, trans, shape):
    # TODO: pytorch3d ?
    return ...



def isContinue(id_1, id_2) -> bool:
    # id_1 > id_2
    u_1, u_2 = id_1, id_2
    assert len(u_1) == 2 and len(u_2) == 2, f'u_1 = {u_1}, u2 == {u_2}'
    return abs(int(u_1[0]) - int(u_2[0])) <= 1

def IsInOneVideo(id_list, idx_1, idx_2) -> bool:
    assert idx_2 > idx_1
    while isContinue(id_list[idx_1], id_list[idx_1+1]) or isContinue(id_list[idx_2-1], id_list[idx_2]):
        idx_1 += 1
        idx_2 -= 1
        if idx_2 <= idx_1: return True
    return False


def getStartEndList(id_list, frame):
    # [start_index, end_index) -> [a, b)
    se_list = []
    pdb.set_trace()
    for i in range(len(id_list)-frame):
        if IsInOneVideo(id_list, i, i + frame - 1):
            se_list.append((i, i + frame))
    # TODO: convert to one line expression after debugging
    return se_list

# root_path = '../download/tumtraf_v2x_cooperative_perception_dataset'

class TumTrafDataset():
    def __init__(self, root_path, frame, mode='train', resolution=(1200, 1920), downsample=8):
        self.frame = frame # video_mamba using
        self.test = os.path.join(root_path, 'test')
        self.train = os.path.join(root_path, 'train')
        self.val = os.path.join(root_path, 'val')
        self.resolution = resolution
        H, W = self.resolution
        self.res = (H, W)
        self.downsample = downsample
        self.downsampler = partial(F.interpolate, size=(H, W), mode='bilinear')

        assert mode in ['train', 'test', 'val']
        self.use = getattr(self, mode)

        self.south1_path = os.path.join(self.use, 'images', 's110_camera_basler_south1_8mm')
        self.south2_path = os.path.join(self.use, 'images', 's110_camera_basler_south2_8mm')
        self.vehicle_path = os.path.join(self.use, 'images', 'vehicle_camera_basler_16mm')

        cfg_base_path = os.path.join(self.train, 'labels_point_clouds', 's110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
        train_list = sorted(os.listdir(cfg_base_path))
        # TODO: need 'src'&'dst'&'transform'&'src img path'&'dst img path'
        self.data_src = []
        self.id_list = []

        for file in tqdm(train_list):
            if file.endswith('.json'):
                with open(os.path.join(cfg_base_path, file), 'r') as f:
                    u = json.load(f)

                key = list(u['openlabel']['frames'])[0]
                file_dict = u['openlabel']['frames'][key]['frame_properties']
                img_list = sorted(file_dict['image_file_names'])
                # pdb.set_trace()
                tmp = img_list[0].split('_')
                id_num = f'{tmp[0]}_{tmp[1]}'

                south1 = os.path.join(self.south1_path, img_list[-1])
                south2 = os.path.join(self.south2_path, img_list[-2])
                vehicle = os.path.join(self.vehicle_path, img_list[1])

                pdb.set_trace()

                if os.path.isfile(south1) and os.path.isfile(south2) and os.path.isfile(vehicle):
                    # TODO: why considerable amount of images in config are missing ?
                    self.id_list.append((tmp[0], tmp[1]))
                    self.data_src.append({
                            'south1': south1,
                            'south2': south2,
                            'vehicle': vehicle,
                            'transform': np.array(
                                file_dict['transforms']['vehicle_lidar_robosense_to_s110_lidar_ouster_south'][
                                    'transform_src_to_dst']['matrix4x4'], dtype=np.float32)
                    })
        
        pdb.set_trace()
        # self.id_list.sort()
        self.start_end_list = getStartEndList(self.id_list, self.frame)

    def convert_dict(self, item, use_south1=True):
        # TODO: choose from 'south1' and 'south2': which is the needed?
        return {
            'vehicle': item['vehicle'], # g.t.
            'intrinsic': item['inf-intrinsic'],
            'video': item['south1' if use_south1 else 'south2'],
            'ray': {
                'pos': item['ray_pos'],
                'dir': item['ray_dir']
            },
        }
        # 'video', ray'

    def __len__(self):
        return len(self.start_end_list)

    def __getitem__(self, idx):
        start, end = self.start_end_list[idx]
        # id_list = self.id_list[start:end]
        dict_list = self.data_src[start:end]
        item_dict = {
            'south1': [], 'south2': [], 'vehicle': [], 'ray_pos': [], 'ray_dir': [], 'inf-intrinsic': []
        }
        pdb.set_trace()
        for u in dict_list:
            item_dict['south1'].append(img2tensor(cv2.imread(u['south1']))[None,:]) # [1 3 h w]
            item_dict['south2'].append(img2tensor(cv2.imread(u['south2']))[None,:]) # [1 3 h w]
            item_dict['vehicle'].append(img2tensor(cv2.imread(u['vehicle']))[None,:]) # [1 3 h w]
            ray = camera2ray(u['transform'], K_veh, self.res)[None,:]
            item_dict['ray_pos'].append(torch.tensor(ray[0], dtype=torch.float32))
            item_dict['ray_dir'].append(torch.tensor(ray[1], dtype=torch.float32))
            item_dict['inf-intrinsic'].append(K_inf)

        item_dict['south1'] = self.downsampler(torch.cat(item_dict['south1'], dim=0))[None, :] # [1 f 3 h w]
        item_dict['south2'] = self.downsampler(torch.cat(item_dict['south2'], dim=0))[None, :] # [1 f 3 h w]
        item_dict['vehicle'] = self.downsampler(torch.cat(item_dict['vehicle'], dim=0))[None, :] # [1 f 3 h w]
        item_dict['ray_pos'] = torch.cat(item_dict['ray_pos'], dim=0)[None, :] # [1 f 3 h w]
        item_dict['ray_dir'] = torch.cat(item_dict['ray_dir'], dim=0)[None, :] # [1 f 3 h w]


        return self.convert_dict(item_dict)


        # TODO: downsample


if __name__ == '__main__':
    dataset = TumTrafDataset(root_path='../../../../../download/tumtraf_v2x_cooperative_perception_dataset', frame=16)
    # pdb.set_trace()
    item = dataset[15]
    print(item.keys())