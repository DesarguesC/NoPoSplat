import json
import cv2
import os, pdb
import open3d as o3d
from copy import copy
import torch.nn as nn
from basicsr.utils import img2tensor, tensor2img
from typing import Optional
from random import randint
import numpy as np
from einops import repeat, rearrange

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



class TumTrafDataset():
    def __init__(self, root_path, frame):
        self.frame = frame # video_mamba using
        self.test = os.path.join(root_path, 'test')
        self.train = os.path.join(root_path, 'train')
        self.val = os.path.join(root_path, 'val')

        self.south1_path = os.path.join(self.train, 'images', 's110_camera_basler_south1_8mm')
        self.south2_path = os.path.join(self.train, 'images', 's110_camera_basler_south2_8mm')
        self.vehicle_path = os.path.join(self.train, 'images', 'vehicle_camera_basler_16mm')

        cfg_base_path = os.path.join(self.train, 'labels_point_clouds', 's110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
        train_list = os.listdir(sorted(cfg_base_path))
        # TODO: need 'src'&'dst'&'transform'&'src img path'&'dst img path'
        self.train_data_src = []
        for file in train_list:
            if file.endswith('.json'):
                with open(os.path.join(file), 'r') as f:
                    u = json.load(f)
                key = list(u['openlabel']['frames'])[0]
                file_dict = u['openlabel']['frames'][key]['frame_properties']
                img_list = sorted(file_dict['image_file_names'])
                self.train_data_src.append(
                    {
                        'south1': os.path.join(self.south1_path, img_list[-1]),
                        'south2': os.path.join(self.south2_path, img_list[-2]),
                        'vehicle': os.path.join(self.vehicle_path, img_list[1]),
                        'transform': np.array(file_dict['transforms']['vehicle_lidar_robosense_to_s110_lidar_ouster_south']['transform_src_to_dst']['matrix4x4'], dtype=np.float32)
                    }
                )

