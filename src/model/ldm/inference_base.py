import argparse
import torch, pdb
from omegaconf import OmegaConf
from typing import NamedTuple
# from torch.nn import partial

from typing import List
from .models.diffusion.ddim import DDIMSampler
from .models.diffusion.plms import PLMSSampler
from .modules.encoders.adapter import Adapter, StyleAdapter, Adapter_light
from .modules.extra_condition.api import ExtraCondition
from .util import fix_cond_shapes, load_model_from_config, read_state_dict
# import mmpose

DEFAULT_NEGATIVE_PROMPT = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                          'fewer digits, cropped, worst quality, low quality'

class Options(NamedTuple):
    train_mode: bool = False
    outdir: str = './outputs/'
    sd_ckpt: str = '../download/v1-5-pruned.ckpt'
    prompt: str = 'a driving scene with high quality, 4K, highly detailed'
    neg_prompt: str = 'poor result, implausible'
    cond_path: str = None
    sampler: str = 'plms'
    steps: int = 50
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae_ckpt: str = None
    adapter_ckpt_path: List[str] = [None, None]
    config: str = './src/model/ldm/configs/stable-diffusion/sd-v1-inference.yaml'
    frame: int = 16
    H: int = 256 # default
    W: int = 256
    C: int = 4
    f: int = 8
    scale: int = 7.5
    seed: int = 42
    cond_weight: List[float] = [1., 1.] # TODO: as dynamic params in training procedure ?
    allow_cond: List[ExtraCondition] = [ExtraCondition.ray, ExtraCondition.feature]

class Train_Options(Options):
    train_mode: bool = True
    prompt: str = 'a driving scene inside the car with high quality, 4K, highly detailed'
    batch_size: int = 8
    epochs: int = 5 # TODO: debug
    num_workers: int = 8 # cpu cores * 2
    auto_resume: bool = True
    config: str = './src/model/ldm/configs/stable-diffusion/sd-v1-train.yaml'
    resume_state_path: str
    name: str = "mamba-feature"
    print_fq: int = 1 # TODO: debug
    H: int = 256
    W: int = 256
    C: int = 4
    f: int = 8
    sample_steps: int = 50
    n_samples: int = 1
    gpus: list = [0,1,2,3] # gpu idx
    local_rank: int = 0
    launcher: str = 'pytorch'

    dec_embed_dim: int = 768
    dec_depth: int = 12
    dec_num_heads: int = 16
    mlp_ratio: int = 4
    norm_im2_in_dec: bool = True
    pos_embed: str = 'cosine'
    decoder_weights_path: str = './pretrained_weights/mixRe10kDl3dv.ckpt'
    device: str = 'cpu' # for parallel, automatically locate


def make_options(train_mode = False):
    return Train_Options() if train_mode else Options()


def get_base_argument_parser() -> argparse.ArgumentParser:
    """get the base argument parser for inference scripts"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outdir',
        type=str,
        help='dir to write results to',
        default=None,
    )

    parser.add_argument(
        '--prompt',
        type=str,
        nargs='?',
        default=None,
        help='positive prompt',
    )

    parser.add_argument(
        '--neg_prompt',
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help='negative prompt',
    )

    parser.add_argument(
        '--sampler',
        type=str,
        default='ddim',
        choices=['ddim', 'plms'],
        help='sampling algorithm, currently, only ddim and plms are supported, more are on the way',
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='number of sampling steps',
    )

    parser.add_argument(
        '--sd_ckpt',
        type=str,
        default='models/sd-v1-4.ckpt',
        help='path to checkpoint of stable diffusion model, both .ckpt and .safetensor are supported',
    )

    parser.add_argument(
        '--vae_ckpt',
        type=str,
        default=None,
        help='vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded',
    )

    parser.add_argument(
        '--adapter_ckpt',
        type=str,
        default=None,
        help='path to checkpoint of adapter',
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/stable-diffusion/sd-v1-inference.yaml',
        help='path to config which constructs SD model',
    )

    parser.add_argument(
        '--max_resolution',
        type=float,
        default=512 * 512,
        help='max image height * width, only for computer with limited vram',
    )

    parser.add_argument(
        '--resize_short_edge',
        type=int,
        default=None,
        help='resize short edge of the input image, if this arg is set, max_resolution will not be used',
    )

    parser.add_argument(
        '--C',
        type=int,
        default=4,
        help='latent channels',
    )

    parser.add_argument(
        '--f',
        type=int,
        default=8,
        help='downsampling factor',
    )

    parser.add_argument(
        '--scale',
        type=float,
        default=7.5,
        help='unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))',
    )

    parser.add_argument(
        '--cond_tau',
        type=float,
        default=1.0,
        help='timestamp parameter that determines until which step the adapter is applied, '
        'similar as Prompt-to-Prompt tau')

    parser.add_argument(
        '--cond_weight',
        type=float,
        default=1.0,
        help='the adapter features are multiplied by the cond_weight. The larger the cond_weight, the more aligned '
        'the generated image and condition will be, but the generated quality may be reduced',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )

    parser.add_argument(
        '--n_samples',
        type=int,
        default=4,
        help='# of samples to generate',
    )

    return parser

def get_sd_models(opt, return_cfg = False):
    """
    build stable diffusion model, sampler
    """
    # SD
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, opt.sd_ckpt, opt.vae_ckpt)
    sd_model = model.to(opt.device)

    # sampler
    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    elif opt.sampler == 'ddim':
        sampler = DDIMSampler(model)
    else:
        raise NotImplementedError
    if return_cfg:
        return sd_model, sampler, config
    else:
        return sd_model, sampler

# Useless
def get_t2i_adapter_models(opt):
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, opt.sd_ckpt, opt.vae_ckpt)
    adapter_ckpt_path = getattr(opt, f'{opt.which_cond}_adapter_ckpt', None)
    if adapter_ckpt_path is None:
        adapter_ckpt_path = getattr(opt, 'adapter_ckpt')
    adapter_ckpt = read_state_dict(adapter_ckpt_path)
    new_state_dict = {}
    for k, v in adapter_ckpt.items():
        if not k.startswith('adapter.'):
            new_state_dict[f'adapter.{k}'] = v
        else:
            new_state_dict[k] = v
    m, u = model.load_state_dict(new_state_dict, strict=False)
    if len(u) > 0:
        print(f"unexpected keys in loading adapter ckpt {adapter_ckpt_path}:")
        print(u)

    model = model.to(opt.device)

    # sampler
    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    elif opt.sampler == 'ddim':
        sampler = DDIMSampler(model)
    else:
        raise NotImplementedError

    return model, sampler


def get_cond_adapter(cond_type: ExtraCondition, frame = 20, device='cuda'):
    # TODO: directly return adapter models
    if cond_type == ExtraCondition.ray:
        return Adapter_light(
            frame = frame,
            cin = frame * 5 * 10,  # frames * 10 * 5 [camera ray map channel]
            channels = [320, 640, 1280, 1280],
            nums_rb = 20,
        ).to(device)
    elif cond_type == ExtraCondition.feature:
        return Adapter(
            frame = frame,
            cin = frame * 10,
            channels=[320, 640, 1280, 1280],
            nums_rb=20,
        ).to(device)
    else:
        raise NotImplementedError('Unrecognized Type')

def get_latent_adapter(opt, train_mode: bool = True, cond_type: List[ExtraCondition] = [None], device = 'cuda'):
    # TODO: refer to app.py to check the usage when calling.
    adapter = {}
    adapter['cond_weight'] = getattr(opt, 'cond_weight', [None]) # list
    adapter['model'] = [get_cond_adapter(cond, frame=opt.frame, device=device) for cond in cond_type]
    if len(adapter['cond_weight']) != len(adapter['model']):
        adapter['cond_weight'] = [1. for i in range(len(adapter['model']))]
    ckpt_path_list = getattr(opt, 'adapter_ckpt_path', [None, None])
    if not train_mode or len(ckpt_path_list) < len(adapter['model']):
        adapter['model'] = [
            adapter['model'][i].load_state_dict(torch.load(ckpt_path_list[i]))
            for i in range(len(adapter['model']))
        ]

    return adapter # dict_keys(['cond_weight', 'model'])


def diffusion_inference(opt, model, sampler, adapter_features, batch_size=1, append_to_context=None):
    # get text embedding
    c = model.get_learned_conditioning([opt.prompt])
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning([opt.neg_prompt])
    else:
        uc = None
    pdb.set_trace()
    c, uc = fix_cond_shapes(model, c, uc) # batch size of c ?

    if not hasattr(opt, 'H'):
        opt.H = 256
        opt.W = 256
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

    samples_latents, _ = sampler.sample(
        S=opt.steps,
        conditioning=c,
        batch_size=batch_size,
        shape=shape,
        verbose=False,
        unconditional_guidance_scale=opt.scale,
        unconditional_conditioning=uc,
        x_T=None,
        features_adapter=adapter_features,
        append_to_context=append_to_context,
        cond_tau=opt.cond_tau,
    )

    x_samples = model.decode_first_stage(samples_latents)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    return x_samples





def train_inference(opt, c, model, sampler, adapter_features, cond_model=None, loss_mode=True, append_to_context=None, *kwargs):

    """
        adapter_features: {
            "mambda": {"dec_feat": ..., "videos": ..., "shapes": ...},
            "ray": <ray-map> # shaped: [b ? ? ?],
        }
    """

    # get text embedding
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning([DEFAULT_NEGATIVE_PROMPT])
    else:
        uc = None
    c, uc = fix_cond_shapes(model, c, uc)

    if not hasattr(opt, 'H'):
        opt.H = 256
        opt.W = 256
        # 最后resize为(1920, 1080)
        print('no------'*10)
    shape = [opt.C, opt.H // opt.factor, opt.W // opt.factor]    # fit the adapter feature

    if (opt.bsize//2)*2 != opt.bsize:
        pdb.set_trace()

    assert (opt.bsize//2)*2 == opt.bsize, 'bad batch size.'
   
    *_, ratios, samples = sampler.sample(
        S=opt.steps,
        conditioning=c,
        batch_size=opt.bsize // 2,
        shape=shape,
        verbose=False,
        unconditional_guidance_scale=opt.scale,
        unconditional_conditioning=uc,
        x_T=None,
        features_adapter=adapter_features,
        append_to_context=append_to_context,
        cond_tau=opt.cond_tau,
        loss_mode=loss_mode,  # need to be trained
        **kwargs

    )
    assert samples != None, '?'
    # print(len(ratios['alphas']), len(samples))
    assert len(ratios['alphas']) == len(samples), 'Fatal: Something went wrong in plms'
    
    for i in range(len(samples)):
        u = model.decode_first_stage(samples[i])
        samples[i] = cond_model(torch.clamp((u + 1.) / 2., min=0., max=1.))

    # seems no need to return an extra ratios matrix
    return samples, ratios
