import os
import argparse
import torch
from third_parties.yacs import CfgNode as CN
import pdb

# pylint: disable=redefined-outer-name
_C = CN()
_C.db = 'zju_mocap'
_C.experiment = 'experiment'
_C.cfg_file_path = ""
_C.debug = False
# "resume" should be train options but we lift it up for cmd line convenience
_C.resume = False
_C.load_pretrained = False
_C.pretrained = ""

# current iteration -- set a very large value for evaluation
_C.eval_iter = 10000000

# for rendering
_C.render_folder_name = ""
_C.render_skip = 1
_C.render_frames = 100

# for data loader
_C.num_workers = 16
_C.sample_visualization = False
def get_cfg_defaults():
    return _C.clone()

def parse_cfg(cfg, args):
    cfg.logdir = os.path.join('experiments', cfg.db, cfg.experiment)
    cfg['validation'] = cfg['progress'].clone()
    cfg.test_white_bg = args.white_bg
    if args.resume:
        cfg['resume'] = True
    
    cfg['evaluate'] = args.test
    if not cfg['evaluate']: # train
        cfg.train.fast_validation = True
        cfg.novel_pose_test = False
        cfg.diff_angle_test = False
        cfg.cliff_estimated_smpl_test = False
    else: # test or validation
        cfg.train.fast_validation = False
        cfg.novel_pose_test = args.novel_pose_test
        cfg.diff_angle_test = args.diff_angle_test
        cfg.cliff_estimated_smpl_test = args.cliff_estimated_smpl_test
        if not 'renderpeople' in cfg.db and args.cliff_estimated_smpl_test:
            print("We are not prepared for cliff estimated smpl test in current test dataset!")
            exit(0)
def determine_primary_gpu(cfg):
    print("------------------ GPU Configurations ------------------")
    cfg.n_gpus = torch.cuda.device_count()
    if cfg.n_gpus > 0:
        all_gpus = list(range(cfg.n_gpus))
        cfg.primary_gpus = [0]
        
        cuda_devices = os.getenv('CUDA_VISIBLE_DEVICES')

    # Check if the environment variable exists
    if cuda_devices is not None:
        print("Manual setting for using GPU number - CUDA_VISIBLE_DEVICES:", cuda_devices)
    
    print(f"Primary GPUs: {cfg.primary_gpus}")
    print("--------------------------------------------------------")

def make_cfg(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/default.yaml')
    cfg.merge_from_file(args.cfg)
    cfg.cfg_file_path = args.cfg
    parse_cfg(cfg, args)
    determine_primary_gpu(cfg)

    return cfg

parser = argparse.ArgumentParser()

# Set configure file for train or test
parser.add_argument("--cfg", required=True, type=str)
parser.add_argument("--resume", action='store_true', help='Resume training process')
parser.add_argument("--test", action='store_true', help='Evaluation mode')
parser.add_argument("--ckpt_path", default='', type=str, help='File path of pre-trained weights')
parser.add_argument("--novel_pose_test", action='store_true', help='Evaluation protocol setting. If True, novel pose synthesis test is performed.')
parser.add_argument("--diff_angle_test", action='store_true', help='Evaluation protocol setting. If True, different angle test is performed.')
parser.add_argument("--cliff_estimated_smpl_test", action='store_true', default=False, help='Evaluation protocol setting. If True, Use cliff estimated smpl in test.')

# Settings for rendering
parser.add_argument("--gt", action='store_true', help='Determine whether output file shows ground truth image when rendering')
parser.add_argument("--alphamap", action='store_true', help='Determine whether output file shows alphamap image when rendering')
parser.add_argument("--white_bg", action='store_true', help='Determine whether output file has white background when rendering')
parser.add_argument("--mode", default='movement', help='Determine render mode')

args = parser.parse_args()
cfg = make_cfg(args)

