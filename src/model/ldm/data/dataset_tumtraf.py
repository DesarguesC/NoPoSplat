import json
import cv2
import os, pdb, torch
import open3d as o3d
from copy import copy
import torch.nn as nn
from basicsr.utils import img2tensor, tensor2img
from typing import Optional
from random import randint
import numpy as np
from utils import posenc_nerf, camera2ray


K_veh = np.array([[2730.3754, 0., 960.], [0., 2730.3754, 600.], [0., 0., 1.]])
# Basler ace acA1920-50gc, 1920×1200, Sony IMX174 with 16 mm lenses
K_inf = np.array([[1365.1877, 0., 960.], [0., 1365.1877, 600.], [0., 0., 1.]])
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



def isContinue(id_1: str, id_2: str) -> bool:
    # id_1 > id_2
    u_1, u_2 = id_1.split('_'), id_2.split('_')
    assert len(u_1) == 2 and len(u_2) == 2, f'u_1 = {u_1}, u2 == {u_2}'
    return (u_1[0] == u_2[0] and str(int(u_1[1]) + 1) == u_2[1]) or (str(int(u_1[0]) + 1) == u_2[0])

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
        if IsInOneVideo(i, i + frame - 1):
            se_list.append((i, i + frame))
    # TODO: convert to one line expression after debugging
    return se_list

# root_path = '../download/tumtraf_v2x_cooperative_perception_dataset'

class TumTrafDataset():
    def __init__(self, root_path, frame):
        self.frame = frame # video_mamba using
        self.test = os.path.join(root_path, 'test')
        self.train = os.path.join(root_path, 'train')
        self.val = os.path.join(root_path, 'val')

        self.resolution = (1200, 1960)

        self.south1_path = os.path.join(self.train, 'images', 's110_camera_basler_south1_8mm')
        self.south2_path = os.path.join(self.train, 'images', 's110_camera_basler_south2_8mm')
        self.vehicle_path = os.path.join(self.train, 'images', 'vehicle_camera_basler_16mm')

        cfg_base_path = os.path.join(self.train, 'labels_point_clouds', 's110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
        train_list = os.listdir(sorted(cfg_base_path))
        # TODO: need 'src'&'dst'&'transform'&'src img path'&'dst img path'
        self.train_data_src = []
        self.id_list = []
        for file in train_list:
            if file.endswith('.json'):
                with open(os.path.join(file), 'r') as f:
                    u = json.load(f)
                key = list(u['openlabel']['frames'])[0]
                file_dict = u['openlabel']['frames'][key]['frame_properties']
                img_list = sorted(file_dict['image_file_names'])
                tmp = img_list[0].split('_')
                id_num = f'{tmp[0]}_{tmp[1]}'
                self.id_list.append(id_num)
                self.train_data_src.append(
                        {
                            id_num : {
                            'south1': os.path.join(self.south1_path, img_list[-1]),
                            'south2': os.path.join(self.south2_path, img_list[-2]),
                            'vehicle': os.path.join(self.vehicle_path, img_list[1]),
                            'transform': np.array(
                                file_dict['transforms']['vehicle_lidar_robosense_to_s110_lidar_ouster_south'][
                                    'transform_src_to_dst']['matrix4x4'], dtype=np.float32)
                        }
                    }
                )
        self.id_list.sort()
        self.start_end_list = getStartEndList(self.id_list, self.frame)

    def convert_dict(self, item, use_south1=True):
        # TODO: choose from 'south1' and 'south2': which is the needed?
        return {
            'video': item['south1' if use_south1 else 'south2'],
            'ray': item['transform'],
        }
        # 'video', ray'

    def __len__(self):
        return len(self.start_end_list)

    def __getitem__(self, idx):
        start, end = self.start_end_list[idx]
        # id_list = self.id_list[start:end]
        dict_list = self.train_data_src[start:end]
        item_dict = {
            'south1': [], 'south2': [], 'vehicle': [], 'transform': []
        }
        for u in dict_list:
            item_dict['south1'].append(img2tensor(cv2.imread(u['south1'])[None,:]))
            item_dict['south2'].append(img2tensor(cv2.imread(u['south2'])[None,:]))
            item_dict['vehicle'].append(img2tensor(cv2.imread(u['vehicle'])[None,:]))
            item_dict['transform'].append(camera2ray(u['transform'], K_inf, self.resolution)[None,:])
        item_dict['south1'] = torch.cat(item_dict['south1'], dim=0)[None, :]
        item_dict['south2'] = torch.cat(item_dict['south2'], dim=0)[None, :]
        item_dict['vehicle'] = torch.cat(item_dict['vehicle'], dim=0)[None, :]
        item_dict['transform'] = torch.cat(item_dict['transform'], dim=0)[None, :]

        return self.convert_dict(item_dict)




