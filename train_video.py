
import logging, yaml, os
import os.path as osp
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           scandir)
from basicsr.utils.options import copy_opt_file, dict2str

from pathlib import Path

import hydra
import torch
import wandb, pdb
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
    opt = make_options(train_mode = True)
    pdb.set_trace()
    cfg = load_yaml_files_recursively(cfg_folder)
    # cfg = load_typed_root_config(cfg)
    cfg.mode = 'val' # useless
    cfg.model.encoder.name = 'videosplat' # ?

    torch.manual_seed(cfg.seed)
    opt = opt._replace(seed=cfg.seed)

    # distributed setting
    init_dist(opt.launcher)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(opt.local_rank)

    # TODO: load data
    pdb.set_trace()
    # opt.frame = cfg.model.encoder.num_frames
    train_dataset = V2XSeqDataset(root_path='../download/V2X-Seq/Sequential-Perception-Dataset/Full Dataset (train & val)', frame=opt.frame)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    pdb.set_trace()
    # Load Model from encoder_videosplat.py
    encoder, _ = get_encoder(cfg.model.encoder)
    # from .model.encoder.encoder_videosplat import EncoderVideoSplat
    # encoder = EncoderVideoSplat(config)
    sd_config = encoder.cfg
    # encoder: encoder_videosplat.py - class EncoderVideoSplat
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # to gpus
    model_video_mamba = torch.nn.parallel.DistributedDataParallel(
        encoder.backbone,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank
    )
    model_ad_ray = torch.nn.parallel.DistributedDataParallel(
        encoder.adapter_dict['ray'],
        device_ids=[opt.local_rank],
        output_device=opt.local_rank
    )
    model_ad_mamba_feat = torch.nn.parallel.DistributedDataParallel(
        encoder.adapter_dict['feature'],
        device_ids=[opt.local_rank],
        output_device=opt.local_rank
    )
    model_sd = torch.nn.parallel.DistributedDataParallel(
        encoder.sd_model,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank
    )

    # optimizer
    params = list(model_video_mamba.parameters()) + list(model_ad_ray.parameters()) + list(model_ad_mamba_feat.parameters())
    optimizer = torch.optim.AdamW(params, lr=sd_config['training']['lr'])

    experiments_root = osp.join('experiments', opt.name)

    # resume state
    resume_state = load_resume_state(opt)
    if resume_state is None:
        mkdir_and_rename(experiments_root)
        start_epoch = 0
        current_iter = 0
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(sd_config))
    else:
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(sd_config))
        resume_optimizers = resume_state['optimizers']
        optimizer.load_state_dict(resume_optimizers)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']

    # copy the yml file to the experiment root
    copy_opt_file(opt.config, experiments_root)

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    for epoch in range(start_epoch, opt.epochs):
        train_dataloader.sampler.set_epoch(epoch)
        # train
        for _, data in enumerate(train_dataloader):
            # TODO: 这里的data要和context一样的结构
            current_iter += 1
            with torch.no_grad():
                c = model_sd.module.get_learned_conditioning(opt.prompt)
                z = model_sd.module.encode_first_stage((data['video'] * 2 - 1.).to(device))
                z = model_sd.module.get_first_stage_encoding(z)

            optimizer.zero_grad()
            model_sd.zero_grad()
            dec_feat, _ = model_video_mamba(data, return_views=True) # only 'video' used
            mamba_feat = model_ad_mamba_feat(dec_feat)
            ray_feat = model_ad_ray(data['ray']) # TODO: * 2 - 1 ???

            features_adapter = opt.cond_weight[0] * mamba_feat + opt.cond_weight[1] * ray_feat

            l_pixel, loss_dict = model_sd(z, c=c, features_adapter=features_adapter)
            l_pixel.backward()
            optimizer.step()

            if (current_iter + 1) % opt.print_fq == 0:
                logger.info(loss_dict)

            # save checkpoint
            rank, _ = get_dist_info()
            if (rank == 0) and ((current_iter + 1) % sd_config['training']['save_freq'] == 0):
                save_filename = f'model_ad_{current_iter + 1}.pth'
                save_path = os.path.join(experiments_root, 'models', save_filename)
                save_dict = {}
                state_dict_list = [
                    model_video_mamba.state_dict(),
                    model_ad_mamba_feat.state_dict(),
                    model_ad_ray.state_dict()
                ]
                model_name = ['video_mamba', 'feature', 'ray']
                for i in range(len(state_dict_list)):
                    for key, param in state_dict_list[i].items():
                        if key.startswith('module.'):  # remove unnecessary 'module.'
                            key = f'{model_name[i]}.{key[7:]}'
                        save_dict[key] = param.cpu()
                torch.save(save_dict, save_path)
                # save state
                state = {'epoch': epoch, 'iter': current_iter + 1, 'optimizers': optimizer.state_dict()}
                save_filename = f'{current_iter + 1}.state'
                save_path = os.path.join(experiments_root, 'training_states', save_filename)
                torch.save(state, save_path)


if __name__ == '__main__':
    main()
