import os
import numpy as np
import torch
from core.train.trainers.human_nerf.lr_updaters.lr_decay import update_lr
from core.nets.create_network import create_network
from core.nets.human_nerf.back_net.generator import Generator, load_pretrained
from core.utils.train_util import remove_batch_axis, cpu_data_to_gpu, Timer, Loss, set_requires_grad
from core.utils.log_util import Logger
from configs import cfg
from torch.utils.tensorboard import SummaryWriter

EXCLUDE_KEYS_TO_GPU = ['frame_name']

class Trainer(object):
    def __init__(self, network, optimizer):
        self.log = Logger()
        self.log.write_config()

        print('\n********** Init Trainer ***********')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.network = network.to(self.device)
        self.network.train()
        
        self.back_net = None
        if cfg.back_net.back_net_on:
            in_chns = 3+24
            self.back_net = Generator(in_chns=in_chns, out_chns=3).to(self.device)
            if cfg.back_net.load_pretrained:
                load_pretrained(self.back_net, cfg.back_net.pretrained)
            set_requires_grad(self.back_net, requires_grad=False)
        
        self.optimizer = optimizer
        
        if cfg.resume:
            self.load_ckpt(cfg.logdir, 'latest')
        else:
            self.iter = 0
            self.save_ckpt(cfg.logdir, 'init')
        
        if cfg.load_pretrained:
            if os.path.exists(cfg.pretrained):
                path, name = os.path.dirname(cfg.pretrained), os.path.basename(cfg.pretrained)[:-1]
                self.load_ckpt(path, name)
                self.iter = 0
                self.save_ckpt(cfg.logdir, 'init')
            else:
                raise FileNotFoundError('File is not exist!')

        self.timer = Timer()
        self.writer = SummaryWriter(cfg.logdir)
        
        self.loss_tracker = {'Loss': []}
        self.loss = Loss(self.device, cfg=cfg)
        if cfg.use_uv_inpainter and cfg.uv_map.uv_adv_loss:
            self.optimizer_F = self.loss.tex_advloss.optimizer_F
        self.time_buf = []
    ## Training 
    def train_begin(self, batch_size):
        assert batch_size == 1

        self.network.train()
        cfg.perturb = cfg.train.perturb

    def train(self, train_dataloader, batch_idx, batch_data):
        self.optimizer.zero_grad()
        if cfg.use_uv_inpainter and cfg.uv_map.uv_adv_loss:
            self.optimizer_F.zero_grad()
        self.train_begin(batch_size=train_dataloader.batch_size)
        
        # only access the first batch as we process one image one time
        batch = remove_batch_axis(batch_data)
        batch['iter_val'] = torch.full((1,), self.iter)
        data = cpu_data_to_gpu(batch, device=self.device, exclude_keys=EXCLUDE_KEYS_TO_GPU)
        bgcolor = data['bgcolor'] / 255.
        
        if not cfg.train.ray_shoot_mode == 'patch':
            target_rgbs = data['target_rgbs']
            target_alpha = data['target_alpha']
            patch_masks = None
            div_indices = None
            
        else:
            target_rgbs = data['target_rgb_patches']
            target_alpha = data['target_alpha_patches']
            patch_masks = data['patch_masks']
            div_indices = data['patch_div_indices']
        
        self.timer.begin()
        net_output = self.network(data, self.back_net)
        
        loss_args = {'lossweights': cfg.train.lossweights,
                    'net_output': net_output,
                    'patch_masks': patch_masks,
                    'target_rgbs': target_rgbs,
                    'div_indices': div_indices,
                    'bgcolor': bgcolor,
                    'target_alpha': target_alpha,
                    'image_height': data['img_height'].item(),
                    'image_width': data['img_width'].item(),
                    'ray_mask': data['ray_mask'],
                    'tight_ray_mask': data['tight_ray_mask']
                    }
        
        if cfg.use_uv_inpainter:
            loss_args.update({'uv_map_gt': data['uv_map_gt']})
            if cfg.uv_map.uv_adv_loss:
                loss_args.update({'uv_map_unpaired_gt': data['adv_tex_gt'],
                                  'inpaintnet': self.network.humannerf.UV_generator.inpainter})


        
        train_loss, loss_dict = self.loss.forward(**loss_args)
        
        train_loss.backward()
        self.optimizer.step()
        if cfg.use_uv_inpainter and cfg.uv_map.uv_adv_loss:
            self.optimizer_F.step()

        self.time_buf.append(self.timer.log())
        
        train_loss_value = train_loss.item()
        self.loss_tracker['Loss'].append(train_loss_value)
        
        for k, v in loss_dict.items():
            if k in self.loss_tracker.keys():
                self.loss_tracker[k].append(v)
            else:
                self.loss_tracker[k] = [v]
        
        if self.iter % cfg.train.log_interval == 0 and self.iter != 0:
            loss_str = f"Loss: {np.mean(np.array(self.loss_tracker['Loss'])):.4f} ["
            for k, v in loss_dict.items():
                loss_mean = np.mean(np.array(self.loss_tracker[k]))
                loss_str += f"{k}: {loss_mean:.4f} "
            loss_str += "]"
            
            for k in list(cfg.train.lossweights.keys()): # Initialize the loss tracker
                self.loss_tracker[k] = []
            self.loss_tracker['Loss'] = []

            mean_fb_prop_time = np.mean(np.array(self.time_buf)) # Initialize the time buffer
            self.time_buf = []
            
            log_str = '[Task {} | Iter {}, {}/{} ({}%) | F/B prop {:.2f} sec | cur_lr {}] {}'
            log_str = log_str.format('Train', 
                                        self.iter, 
                                        batch_idx * cfg.train.batch_size, 
                                        len(train_dataloader.dataset),
                                        int(100. * batch_idx / len(train_dataloader)), 
                                        mean_fb_prop_time, self.optimizer.param_groups[0]['lr'], loss_str
                                        )
            print(log_str)
            for k, v in loss_dict.items():
                self.writer.add_scalar(tag='train/{}'.format(k), scalar_value=v, global_step=self.iter)
            self.log.write(log_str)
            
        self.iter += 1
        update_lr(self.optimizer, self.iter)
        
    def reload_network(self):
        self.network = create_network().to(self.device)
        self.save_ckpt(cfg.logdir, 'init')
        self.iter = 0

    ## Utils
    def save_ckpt(self, path, name):
        path_ = os.path.join(path, name+'.tar')
        print(f"Save checkpoint to {path_} ...")
        state_dict = {
            'iter': self.iter,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state_dict, path_)

    def load_ckpt(self, path, name):
        path_ = os.path.join(path, name+'.tar')
        print(f"Load checkpoint from {path_} ...")
        
        ckpt = torch.load(path_)
        self.iter = ckpt['iter']
        self.network.load_state_dict(ckpt['network'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer'])