import torch
import numpy as np
from core.nets.create_network import create_network
from core.train.create_components import create_progress
from core.nets.human_nerf.back_net.generator import Generator, load_pretrained
from core.utils.train_util import set_requires_grad
from core.utils.misc import count_parameters
from configs import cfg, args
import pdb
import random

def load_ckpt(model, path, device):
    print(f"Load checkpoint from {path} ...")
    ckpt = torch.load(path, map_location=device)
    cur_iter = ckpt['iter']
    model.load_state_dict(ckpt['network'], strict=False)
    return model, cur_iter

def main():
    cfg.evaluate = True
    cfg.novel_pose_test = args.novel_pose_test
    cfg.diff_angle_test = args.diff_angle_test
    
    random.seed(5000)
    np.random.seed(5000)
    torch.manual_seed(100000)
    torch.cuda.manual_seed(100000)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_network()
    model = model.to(device)
    ckptpath = args.ckpt_path
    model, cur_iter = load_ckpt(model, ckptpath, device)
    if cfg.back_net.back_net_on:
        in_chns = 27
        back_net = Generator(in_chns=in_chns, out_chns=3).to(device)
        if cfg.back_net.load_pretrained:
            load_pretrained(back_net, cfg.back_net.pretrained)
        set_requires_grad(back_net, requires_grad=False)
    else:
        back_net = None
    print("Loaded network")
    # print('Number of parameters | BackNet: {} (total, trainable)'.format(count_parameters(back_net)))
    full_tester = create_progress(fast_validation=False, test_mode=True)
    full_tester.set_device(device)
    print("Loaded Progress, Test Module")
    
    # if not cfg.diff_angle_test:
        # full_tester.render_progress(model, cur_iter, back_net=back_net)
    full_tester.validate(model, cur_iter, writer=None, save_samples=True, back_net=back_net)
    
    print("Testing complete")

if __name__ == '__main__':
    main()