import json
import cv2
import os, pdb, torch
from tqdm import tqdm
from basicsr.utils import img2tensor, tensor2img
import numpy as np
from einops import repeat, rearrange
from .utils import positional_encode, camera2ray, load_v2x_intrinsics, load_v2x_transform
from torch.nn import functional as F
from functools import partial

# root_path = '../download/V2X-Seq/Sequential-Perception-Dataset/Full Dataset (train & val)/'

def ab64(u: int) -> int:
    return 64 * int(u/64 + 0.5)


class V2XSeqDataset():

    def load_transform_i2v_matrix2camera(self, trans_path_list: list):
        # infrastructure side
        C_inf_2_lidar = np.linalg.inv(load_v2x_transform(trans_path_list[0]))
        lidar_2_world = load_v2x_transform(trans_path_list[1])
        # vehicle side
        world_2_novatel = np.linalg.inv(load_v2x_transform(trans_path_list[2]))
        novatel_2_lidar = np.linalg.inv(load_v2x_transform(trans_path_list[3]))
        lidar_2_C_veh = load_v2x_transform(trans_path_list[4])

        transform_i2v = C_inf_2_lidar @ lidar_2_world @ world_2_novatel @ novatel_2_lidar @ lidar_2_C_veh

        veh_intrinsic = load_v2x_intrinsics(trans_path_list[5])
        pos, dir = camera2ray(transform_i2v, veh_intrinsic, resolution=self.res)
        return (torch.tensor(pos, dtype=torch.float32), torch.tensor(dir, dtype=torch.float32))

    def __init__(self, root_path, frame, resolution=(1080,1920), downsample=4, train_res=(256,256)):
        self.frame = frame
        self.root_path = root_path
        self.config_path = os.path.join(root_path, 'V2X-Seq-SPD', 'cooperative/data_info.json')

        self.resolution = resolution
        H, W = resolution
        self.downsample = downsample
        # self.res = (ab64(H // downsample), ab64(W // downsample))
        # TODO: set H = W ?
        # pdb.set_trace()
        # self.res = (self.res[0], self.res[0])
        self.res = train_res

        # downsample when return items

        vehicle_image_path = os.path.join(root_path, 'V2X-Seq-SPD-vehicle-side-image')
        infrastructure_image_path = os.path.join(root_path, 'V2X-Seq-SPD-infrastructure-side-image')
        self.vehicle_config_path = os.path.join(root_path, 'V2X-Seq-SPD', 'vehicle-side/data_info.json')
        self.infrastructure_config_path = os.path.join(root_path, 'V2X-Seq-SPD', 'infrastructure-side/data_info.json')

        H, W = train_res
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
        vehicle_images_list = sorted(os.listdir(vehicle_image_path))
        infrastructure_images_list = sorted(os.listdir(infrastructure_image_path))
        vehicle_images, infrastructure_images, calibration, veh_transform, trans = {}, {}, {}, {}, {}
        """
            {
                <sequence>: [<veh/inf image>]
            }
            camera: vehicle camera intrinsics | transform camera extrinsics
        """
        with open(self.config_path, 'r') as file:
            data_info = json.load(file)
        frames = [(u['vehicle_frame'], u['infrastructure_frame'], u['vehicle_sequence'], u['infrastructure_sequence']) for u in data_info]

        pdb.set_trace()
        for f in tqdm(frames):
            assert f[2] == f[3], f'unequal sequence number! veh_frame = {f[0]}, inf_veh = {f[1]}'

            veh_path = veh_search_dict[f[0]]
            inf_path = inf_search_dict[f[1]]

            calib_list = [
                           os.path.join(root_path, 'V2X-Seq-SPD/infrastructure-side', inf_path[0]),
                           os.path.join(root_path, 'V2X-Seq-SPD/infrastructure-side', inf_path[1]),
                           os.path.join(root_path, 'V2X-Seq-SPD/vehicle-side', veh_path[0]),
                           os.path.join(root_path, 'V2X-Seq-SPD/vehicle-side', veh_path[1]),
                           os.path.join(root_path, 'V2X-Seq-SPD/vehicle-side', veh_path[2]),
                           os.path.join(root_path, 'V2X-Seq-SPD/vehicle-side', veh_path[-1]), # veh intrinsic
                           os.path.join(root_path, 'V2X-Seq-SPD/infrastructure-side', inf_path[-1]),  # inf intrinsic
                           ]

            if f[0] + '.jpg' in vehicle_images_list and f[1] + '.jpg' in infrastructure_images_list:
                if f[2] in vehicle_images:
                    vehicle_images[f[2]].append(os.path.join(vehicle_image_path, f[0] + '.jpg'))
                    infrastructure_images[f[2]].append(os.path.join(infrastructure_image_path, f[1] + '.jpg'))
                    calibration[f[2]].append(calib_list)

                else:
                    vehicle_images[f[2]] = [os.path.join(vehicle_image_path, f[0] + '.jpg')]
                    infrastructure_images[f[2]] = [os.path.join(infrastructure_image_path, f[1] + '.jpg')]
                    calibration[f[2]] = [calib_list]

            else:
                continue


        self.vehicle_list = []
        self.infrastructure_list = []
        self.inf_intrinsic_path_list = []
        self.transform_path_list = []
        for (k, v) in vehicle_images.items():
            L = len(v)
            for i in range(L-frame):
                self.vehicle_list.append(v[i:i+frame])
                self.infrastructure_list.append(infrastructure_images[k][i:i+frame])
                self.inf_intrinsic_path_list.append([u[-1] for u in calibration[k][i:i+frame]])
                self.transform_path_list.append([u[:-1] for u in calibration[k][i:i+frame]])

    def convert_item(self, item):
        pdb.set_trace()
        rays = torch.cat([
                item['ray_pos'][:, :2], .5 * (item['ray_pos'][:, 2] + item['ray_dir'][:, 0])[None, :], item['ray_dir'][:, 1:]
            ], dim=1) # [f 5 h w]
        rays = repeat(rays[None, :], '1 ... -> n ...', n = 10) # mamba windows length
        return {
            'video': item['video'] / 255.,      # [f 3 h w]
            'intrinsic': item['intrinsic'],     # [f 3 3]
            'vehicle': item['vehicle'] / 255.,  # [f 3 h w]
            # g.t.
            'ray': rearrange(rays, 'n f c h w -> (n f c) h w')
        }

    def __len__(self):
        return len(self.vehicle_list)

    def __getitem__(self, idx):
        item_dict = {}
        item_dict['video'] = torch.cat([img2tensor(cv2.imread(u))[None,:] for u in self.infrastructure_list[idx]], dim=0)
        item_dict['video'] = self.downsampler(item_dict['video']) # [f 3 h w]
        item_dict['vehicle'] = torch.cat([img2tensor(cv2.imread(u))[None, :] for u in self.vehicle_list[idx]], dim=0)
        item_dict['vehicle'] = self.downsampler(item_dict['vehicle']) # [f 3 h w]
        intrinsic_list = [load_v2x_intrinsics(u) for u in self.inf_intrinsic_path_list[idx]]
        ray_map_list = [self.load_transform_i2v_matrix2camera(u) for u in self.transform_path_list[idx]]

        item_dict['ray_pos'], item_dict['ray_dir'] = [], []
        for j in range(len(intrinsic_list)):
            item_dict['ray_pos'].append(ray_map_list[j][0][None, :])
            item_dict['ray_dir'].append(ray_map_list[j][1][None, :])

        item_dict['ray_pos'] = rearrange(torch.cat(item_dict['ray_pos'], dim=0) / 255., 'f h w c -> f c h w') / 255. # [f 3 h w]
        item_dict['ray_dir'] = rearrange(torch.cat(item_dict['ray_dir'], dim=0), 'f h w c -> f c h w') # [f 3 h w]
        # TODO: read
        item_dict['intrinsic'] = torch.cat([torch.tensor(u[None, :], dtype=torch.float32) for u in intrinsic_list]) # [f 3 3]

        return self.convert_item(item_dict)

if __name__ == '__main__':
    dataset = V2XSeqDataset(root_path='../../../../../download/V2X-Seq/Sequential-Perception-Dataset/Full Dataset (train & val)', frame=16)
    pdb.set_trace()
    item = dataset[15]
    print(item.keys())

    print('Done.')
