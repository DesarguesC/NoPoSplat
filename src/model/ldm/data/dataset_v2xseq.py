import json
import cv2
import os, pdb, torch
from basicsr.utils import img2tensor, tensor2img
import numpy as np
from einops import repeat, rearrange
from utils import positional_encode, camera2ray, load_v2x_intrinsics, load_v2x_transform
from torch.nn import functional as F
from functools import partial

# root_path = '../download/V2X-Seq/Sequential-Perception-Dataset/Full Dataset (train & val)/'
# TODO: prompt -> "inside the car"

class V2XSeqDataset():
    def __init__(self, root_path, frame, resolution=(1080,1920), downsample=4):
        self.frame = frame
        self.root_path = root_path
        self.config_path = os.path.join(root_path, 'V2X-Seq-SPD', 'cooperative/data_info.json')

        self.resolution = resolution
        self.downsample = downsample
        # downsample when return items

        vehicle_image_path = os.path.join(root_path, 'V2X-Seq-SPD-vehicle-side-image')
        infrastructure_image_path = os.path.join(root_path, 'V2X-Seq-SPD-infrastructure-side-image')
        self.vehicle_config_path = os.path.join(root_path, 'V2X-Seq-SPD', 'vehicle-side/data_info.json')
        self.infrastructure_config_path = os.path.join(root_path, 'V2X-Seq-SPD', 'infrastructure-side/data_info.json')

        H, W = self.resolution
        self.downsampler = partial(F.interpolate, size=(H, W), mode='bilinear')
        # img-id ~ calib path list
        with open(self.infrastructure_config_path, 'r') as f:
            inf_cfg = json.load(f)
            inf_cfg = sorted(inf_cfg, key = lambda x: x['frame_id'])
        with open(self.vehicle_config_path, 'r') as f:
            veh_cfg = json.load(f)
            veh_cfg = sorted(veh_cfg, key = lambda x: x['frame_id'])

        inf_search_dict = {u['frame_id']: (u['calib_virtuallidar_to_camera_path'], u['calib_virtuallidar_to_world_path'], u['calib_camera_intrinsic_path']) for u in inf_cfg}
        veh_search_dict = {u['frame_id']: (u['calib_novatel_to_world_path'], u['calib_lidar_to_novatel_path'], u['calib_lidar_to_camera_path'], u['calib_camera_intrinsic_path']) for u in veh_cfg}

        vehicle_images_list = os.listdir(vehicle_image_path).sort()
        infrastructure_images_list = os.listdir(infrastructure_image_path).sort()

        vehicle_images, infrastructure_images, inf_intrinsics, veh_ray_maps = {}, {}, {}, {}
        """
            {
                <sequence>: [<veh/inf image>]
            }
            camera: vehicle camera intrinsics | transform camera extrinsics
        """
        pdb.set_trace()
        with open(self.config_path, 'r') as file:
            data_info = json.load(file)
        frames = [(u['vehicle_frame'], u['infrastructure_frame'], u['vehicle_sequence'], u['infrastructure_sequence']) for u in data_info]
        for f in frames:
            assert f[2] == f[3], f'unequal sequence number! veh_frame = {f[0]}, inf_veh = {f[1]}'

            veh_path = veh_search_dict[f[0]]
            inf_path = inf_search_dict[f[1]]

            inf_intrinsic = load_v2x_intrinsics(os.path.join(root_path, 'V2X-Seq-SPD/infrastructure-side', inf_path[-1]))
            veh_intrinsic = load_v2x_intrinsics(os.path.join(root_path, 'V2X-Seq-SPD/vehicle', veh_path[-1]))
            # infrastructure side
            C_inf_2_lidar = np.linalg.inv(load_v2x_transform(os.path.join(root_path, 'V2X-Seq-SPD/infrastructure-side', inf_path[0])))
            lidar_2_world = load_v2x_transform(os.path.join(root_path, 'V2X-Seq-SPD/infrastructure-side', inf_path[1]))
            # vehicle side
            world_2_novatel = np.linalg.inv(load_v2x_transform(os.path.join(root_path, 'V2X-Seq-SPD/vehicle-side', veh_path[0])))
            novatel_2_lidar = np.linalg.inv(load_v2x_transform(os.path.join(root_path, 'V2X-Seq-SPD/vehicle-side', veh_path[1])))
            lidar_2_C_veh = load_v2x_transform(os.path.join(root_path, 'V2X-Seq-SPD/vehicle-side', veh_path[2]))

            transform_i2v = C_inf_2_lidar @ lidar_2_world @ world_2_novatel @ novatel_2_lidar @ lidar_2_C_veh


            if f[0] + '.jpg' in vehicle_images_list and f[1] + '.jpg' in infrastructure_images_list:
                if f[2] in vehicle_images:
                    vehicle_images[f[2]].append(os.path.join(vehicle_image_path, f[0] + '.jpg'))
                    infrastructure_images[f[2]].append(os.path.join(infrastructure_image_path, f[1] + '.jpg'))
                    inf_intrinsics[f[2]].append(inf_intrinsic)
                    veh_ray_maps[f[2]].append(camera2ray(transform_i2v, veh_intrinsic, resolution))

                else:
                    vehicle_images[f[2]] = [os.path.join(vehicle_image_path, f[0] + '.jpg')]
                    infrastructure_images[f[2]] = [os.path.join(infrastructure_image_path, f[1] + '.jpg')]
                    inf_intrinsics[f[2]] = [inf_intrinsic]
                    veh_ray_maps[f[2]] = [camera2ray(transform_i2v, veh_intrinsic, resolution)]
            else:
                continue

        self.vehicle_list = []
        self.infrastructure_list = []
        self.inf_intrinsic_list = []
        self.ray_map_list = [] # ray of transform
        for (k, v) in vehicle_images:
            L = len(v)
            for i in range(L-frame):
                self.vehicle_list.append(vehicle_images[i:i+frame])
                self.infrastructure_list.append(infrastructure_images[i:i+frame])
                self.inf_intrinsic_list.append(inf_intrinsics[i:i+frame])
                self.ray_map_list.append(veh_ray_maps[i:i+frame])


    def __len__(self):
        return len(self.vehicle_list)

    def __getitem__(self, idx):
        item_dict = {}
        item_dict['video'] = torch.cat([img2tensor(cv2.imread(u))[None,:] for u in self.infrastructure_list[idx]], dim=0) # [f 3 h w]
        item_dict['ray'] = torch.cat([u[None, :] for u in self.ray_map_list[idx]])[None, :] # [f h w 3]
        item_dict['intrinsic'] = torch.cat([u[None, :] for u in self.inf_intrinsic_list[idx]])[None, :] # [f 3 3]
        item_dict['vehicle'] = torch.cat([img2tensor(cv2.imread(u))[None,:] for u in self.vehicle_list[idx]], dim=0) # [f 3 h w]

        item_dict['video'] = self.downsampler(item_dict['video'])[None, :]
        item_dict['ray'] = self.downsampler(item_dict['ray'])[None, :]
        item_dict['vehicle'] = self.downsampler(item_dict['vehicle'])[None, :]

        return item_dict

if __name__ == '__main__':
    dataset = V2XSeqDataset(root_path='../../../../../download/V2X-Seq/Sequential-Perception-Dataset/Full Dataset (train & val)', frame=16)
    pdb.set_trace()
    item = dataset[15]
    print(item.keys())


