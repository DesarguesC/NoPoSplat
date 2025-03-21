import logging, yaml, os
import os.path as osp
from accelerate import Accelerator
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           scandir)
from basicsr.utils.options import copy_opt_file, dict2str
from einops import repeat, rearrange
import argparse
import hydra
import torch
import torch.distributed as dist
from datetime import datetime
import wandb, pdb, itertools
import signal
from colorama import Fore
# from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf

# Configure beartype and jaxtyping.
# with install_import_hook(
#     ("src",),
#     ("beartype", "beartype"),
# ):

from src.model.ldm.data import V2XSeqDataset # , TumTrafDataset
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only
from src.config import load_typed_root_config
from src.model.ldm import *
from src.model.encoder import get_encoder
# from src.model.model_wrapper import ModelWrapper

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

@master_only
def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(path, 'models'))
    os.makedirs(osp.join(path, 'training_states'))
    os.makedirs(osp.join(path, 'visualization'))


def load_resume_state(opt):
    resume_state_path = None
    if opt.auto_resume:
        state_path = osp.join('experiments', opt.name, 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt.resume_state_path = resume_state_path

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
    return resume_state

def convert_to_dictconfig(obj):
    # 如果是字典类型，将其中的每个项都转换为DictConfig
    if isinstance(obj, dict):
        return OmegaConf.create({key: convert_to_dictconfig(value) for key, value in obj.items()})
    # 如果是列表类型，递归地处理列表中的每个元素
    elif isinstance(obj, list):
        return [convert_to_dictconfig(item) for item in obj]
    # 如果是其他类型，直接返回
    return obj

def recursive_merge(config, new_config):
    """
    Recursively merge new_config into config. If fields are the same,
    new_config fields will be merged into existing ones as subfields.
    """
    for key, value in new_config.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            # If both values are dictionaries, recursively merge them
            recursive_merge(config[key], value)
        else:
            # Otherwise, simply set the value
            config[key] = value

def load_yaml_files_recursively(folder_path):
    config = OmegaConf.create()  # 创建一个空的DictConfig对象
    # pdb.set_trace()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):  # 检查文件扩展名
                file_path = os.path.join(root, file)
                # pdb.set_trace()
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 解析yaml文件并转为DictConfig
                        file_config = OmegaConf.create(yaml.safe_load(f))
                        config.merge_with(file_config)  # 合并到config中
                        # recursive_merge(config, file_config)
                except (IOError, yaml.YAMLError) as e:
                    print(f"Error loading {file_path}: {e}")
                    continue  # 如果发生错误，跳过该文件

    # 将config中的所有子对象递归转换为DictConfig
    config = convert_to_dictconfig(config)
    return config


from pathlib import Path


def read_config_folder(config_path: Path) -> OmegaConf:
    """
    递归读取一个config文件夹，返回一个DictConfig类型的变量。
    :param config_path: 配置文件夹路径
    :return: OmegaConf对象
    """
    # 判断路径是否是文件夹
    if config_path.is_dir():
        # 创建一个空的DictConfig对象
        config_dict = {}

        # 遍历文件夹中的每个项目（文件夹或文件）
        for item in config_path.iterdir():
            # 如果是子文件夹，递归调用该函数
            if item.is_dir():
                config_dict[item.name] = read_config_folder(item)
            # 如果是YAML文件，加载其内容
            elif item.suffix == '.yaml':
                with open(item, 'r', encoding='utf-8') as f:
                    # 加载yaml文件内容到字典
                    yaml_content = yaml.safe_load(f)
                    # 使用文件名（不包括扩展名）作为键，值为yaml内容
                    config_dict[item.stem] = yaml_content

        # 使用OmegaConf包装字典，返回一个DictConfig对象
        return OmegaConf.create(config_dict)
    else:
        raise ValueError(f"{config_path} 不是一个有效的文件夹路径")



def main(cfg_folder: str = './config'):

    opt = make_options(train_mode = False)
    cfg = load_yaml_files_recursively(cfg_folder)
    cfg.mode = 'train' # useless
    cfg.model.encoder.name = 'videosplat' # ?

    torch.manual_seed(cfg.seed)
    opt = opt._replace(seed=cfg.seed)
    # pdb.set_trace()

    # TODO: load data
    # pdb.set_trace()
    inf_dataset = V2XSeqDataset(root_path='../download/V2X-Seq/Sequential-Perception-Dataset/Full Dataset (train & val)', frame=opt.frame, cut_down_scale=1)
    # Load Model from encoder_videosplat.py
    encoder, _ = get_encoder(cfg.model.encoder, args=opt)
    # torch.cuda.empty_cache()

    encoder.adapter_dict['model'][0] = encoder.adapter_dict['model'][0].cuda()
    encoder.adapter_dict['model'][1] = encoder.adapter_dict['model'][1].cuda()
    encoder.sd_model = encoder.sd_model.cuda()
    encoder.backbone = encoder.backbone.cuda()

    encoder.backbone.eval()
    encoder.adapter_dict['model'][0].eval()
    encoder.adapter_dict['model'][1].eval()
    encoder.sd_model.eval()

    A = {}
    for (k,v) in inf_dataset[0].items():
        A[k] = v.to('cuda')

    output = encoder(A)
    pdb.set_trace()
    print(f'output.shape = {output.shape}')




if __name__ == '__main__':
    main()
