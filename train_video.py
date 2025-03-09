import logging, yaml, os
import os.path as osp
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration

from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           scandir)
from basicsr.utils.options import copy_opt_file, dict2str
from einops import repeat, rearrange
from transformers import get_scheduler
import argparse
import hydra
import torch
import torch.distributed as dist
from datetime import datetime
import wandb, pdb, itertools
import signal
from colorama import Fore
from torch.utils.data import DataLoader
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

    # accelerator = Accelerator()
    accelerator = Accelerator(
        mixed_precision='bf16',  # Enable automatic mixed precision
        gradient_accumulation_steps=1,  # Adjust if needed
        log_with="wandb",  # If you're using wandb
    )
    
    opt = make_options(train_mode = True)
    cfg = load_yaml_files_recursively(cfg_folder)
    cfg.mode = 'train' # useless
    cfg.model.encoder.name = 'videosplat' # ?

    torch.manual_seed(cfg.seed)
    opt = opt._replace(seed=cfg.seed)

    train_dataset = V2XSeqDataset(root_path='../download/V2X-Seq/Sequential-Perception-Dataset/Full Dataset (train & val)', frame=opt.frame, cut_down_scale=1)
    # Load Model from encoder_videosplat.py
    encoder, _ = get_encoder(cfg.model.encoder, args=opt)
    encoder.backbone = encoder.backbone.cuda()
    encoder.backbone.train()
    encoder.adapter_dict['model'][0].train()
    encoder.adapter_dict['model'][1].train()
    backbone_model = encoder.backbone.cuda() # VideoMamba
    model_sd = encoder.sd_model
    model_sd.train()

    sd_config = encoder.sd_cfg

    class ModelWrapper(torch.nn.Module):
        def __init__(self, backbone, adapter_0, adapter_1):
            super().__init__()
            # self.backbone = backbone
            self.adapter_0 = adapter_0
            self.adapter_1 = adapter_1


        def forward(self, x, item):
            # dec_feat, _ = self.backbone(context=x, return_views=True)
            mamba_feat = self.adapter_0(x)
            ray_feat = self.adapter_1(rearrange(item['ray'], 'b (c f) h w -> (b f) c h w', f=opt.frame))
            
            # Add shape validation
            assert len(mamba_feat) == len(ray_feat), "Feature list length mismatch"
            features_adapter = [
                opt.cond_weight[0] * mamba_feat[i] + opt.cond_weight[1] * ray_feat[i] 
                for i in range(len(mamba_feat))
            ]
            
            return features_adapter
    
    # Check and fix device placement
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)  # TODO: with "train_dataset"
    v2x_wrapper = ModelWrapper(encoder.backbone, encoder.adapter_dict['model'][0], encoder.adapter_dict['model'][1])
    optimizer = torch.optim.AdamW(v2x_wrapper.parameters(), lr=sd_config['training']['lr'])
    resume_state = load_resume_state(opt)
    start_epoch = 0 if resume_state is None else resume_state['epoch']
    num_training_steps = (opt.epochs - start_epoch + 1) * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    experiments_root = osp.join('experiments', opt.name)
    # resume state
    if resume_state is None:
        mkdir_and_rename(experiments_root)
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
        pdb.set_trace()
        resume_optimizers = resume_state['optimizers']
        optimizer.load_state_dict(resume_optimizers)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        current_iter = resume_state['iter']

    # pdb.set_trace()
    train_dataloader, v2x_generator, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader, v2x_wrapper, optimizer, lr_scheduler
    )

    # copy the yml file to the experiment root
    copy_opt_file(opt.config, experiments_root)
    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    move_to_gpu = lambda data: {k: v.cuda() for (k,v) in data.items()}
    torch.cuda.empty_cache()

    optimizer.zero_grad()
    model_sd.zero_grad()
    backbone_model.zero_grad()

    for epoch in range(start_epoch, opt.epochs): # TODO: check 'c' shape
        # train_dataloader.sampler.set_epoch(epoch)
        # train
        for _, data in enumerate(train_dataloader): # first check: train_dataset[0]
            # TODO: 这里的data要和context一样的结构
            current_iter += 1
            data = move_to_gpu(data)

            with torch.no_grad():
                # video = rearrange(data['video'], 'b f c h w -> (b f) c h w')
                c = model_sd.get_learned_conditioning([opt.prompt])
                c = repeat(c, '1 ... -> b ...', b = (opt.frame * opt.batch_size))
                vehicle = rearrange(data['vehicle'], 'b f c h w -> (b f) c h w')
                z = model_sd.encode_first_stage((vehicle * 2 - 1.).cuda(non_blocking=True)) # not ".to(device)"
                z = model_sd.get_first_stage_encoding(z) # padding the noise
                dec_feat, _ = backbone_model(context=data, return_views=True)

            # TODO: 这里需要修改
            adapter_features = v2x_generator(dec_feat, data)
            l_pixel, loss_dict = model_sd(z, c=c, features_adapter=adapter_features)

            accelerator.backward(l_pixel)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            model_sd.zero_grad()
            backbone_model.zero_grad()
            torch.cuda.empty_cache()

            if (current_iter + 1) % opt.print_fq == 0:
                logger.info(loss_dict)

            # save checkpoint
            
            if accelerator.is_main_process and ((current_iter + 1) % sd_config['training']['save_freq'] == 0):
            # if rank == 0: # TODO: Debug
            #     pdb.set_trace()
                save_filename = f'v2x_generator_{current_iter + 1}.pth'
                save_path = os.path.join(experiments_root, 'models', save_filename)
                save_dict = {}
                state_dict_list = [
                    v2x_generator.state_dict()
                ]
                model_name = ['video_mamba', 'feature', 'ray'] # backbone & adapter_0 & adapter_0
                for i in range(len(state_dict_list)):
                    for key, param in state_dict_list[i].items():
                        if key.startswith('module.'):  # remove unnecessary 'module.'
                            key = f'{model_name[0]}.{key[16:]}' if key[7:].startswith('backbone') else \
                                  f'{model_name[1]}.{key[17:]}' if key[7:].startswith('adapter_0') else \
                                  f'{model_name[2]}.{key[17:]}' if key[7:].startswith('adapter_1') else \
                                  None
                        save_dict[key] = param.cpu()
                # pdb.set_trace()
                torch.save(save_dict, save_path)
                # save state
                state = {'epoch': epoch, 'iter': current_iter + 1, 'optimizers': optimizer.state_dict()}
                save_filename = f'{current_iter + 1}.state'
                save_path = os.path.join(experiments_root, 'training_states', save_filename)
                torch.save(state, save_path)


if __name__ == '__main__':
    main()
