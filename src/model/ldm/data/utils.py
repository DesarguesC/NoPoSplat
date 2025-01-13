# -*- coding: utf-8 -*-

import cv2
import numpy as np
import visu3d as v3d
from einops import repeat, rearrange

from torchvision.transforms import transforms
from torchvision.transforms.functional import to_tensor
from transformers import CLIPProcessor

from basicsr.utils import img2tensor

def posenc_nerf(x, min_deg=0, max_deg=15):
    """Concatenate x and its positional encodings, following NeRF."""
    if min_deg == max_deg:
        return x
    scales = np.array([2**i for i in range(min_deg, max_deg)])
    # print(f'scales.shape = {scales.shape}')
    xb = np.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    # print(f'xb.shape = {xb.shape}')
    emb = np.sin(np.concatenate([xb, xb + np.pi / 2.], axis=-1)) # sin(2^ix), cos(2^ix), ...
    # print(f'emb.shape = {emb.shape}')
    return np.concatenate([x, emb], axis=-1)

def camera2ray(transform, intrinsic, resolution):
    R, T = transform[0:3, 0:3], transform[0:3,-1]
    w2c = v3d.Transform(R=R, t=T)
    cam_spec = v3d.PinholeCamera(resolution=resolution, K=intrinsic)
    rays = v3d.Camera(spec=cam_spec, world_from_cam=w2c).rays()
    ray_map = np.asarray(rays) # -> resolution = (H, W)
    # TODO: for optimization
    # pos_emb_pos = posenc_nerf(rays.pos, min_deg=0, max_deg=15) # (H, W, 93)
    return rearrange(repeat(ray_map[None,:], '1 ... -> c ...', c = 3), 'c h w -> h w c')


class AddCannyFreezeThreshold(object):

    def __init__(self, low_threshold=100, high_threshold=200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, sample):
        # sample['jpg'] is PIL image
        x = sample['jpg']
        img = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
        canny = cv2.Canny(img, self.low_threshold, self.high_threshold)[..., None]
        sample['canny'] = img2tensor(canny, bgr2rgb=True, float32=True) / 255.
        sample['jpg'] = to_tensor(x)
        return sample


class AddStyle(object):

    def __init__(self, version):
        self.processor = CLIPProcessor.from_pretrained(version)
        self.pil_to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        # sample['jpg'] is PIL image
        x = sample['jpg']
        style = self.processor(images=x, return_tensors="pt")['pixel_values'][0]
        sample['style'] = style
        sample['jpg'] = to_tensor(x)
        return sample


# 需要重新定义一个损失函数？
class Loss():
    def __inti__(self, pri_ad, sec_ad):
        self.pri = pri_ad
        self.sec = sec_ad



