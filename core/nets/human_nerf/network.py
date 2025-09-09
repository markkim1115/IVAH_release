import torch
import torch.nn as nn
from core.nets.human_nerf.image_encoder import ImageEncoder
from core.nets.human_nerf.human_nerf import HumanNeRF
from core.utils.camera_util import sample_points_along_ray
from configs import cfg

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.img_encoder = ImageEncoder(model_name=cfg.image_enc.backbone)
        self.humannerf = HumanNeRF()
        
        # print('Number of parameters | Image Encoder: {} (total, trainable)'.format(count_parameters(self.img_encoder)))
    def process_humannerf(self, data:dict, 
                          rays_o:torch.Tensor,
                          rays_d:torch.Tensor, 
                          near:torch.Tensor, 
                          far:torch.Tensor,
                          ):
        z_vals, pts = sample_points_along_ray(rays_o, rays_d, near, far, cfg.N_samples, cfg.perturb)

        ret = self.humannerf(rays_o, rays_d, pts, z_vals, **data)
        
        render_outs = {}
        for k in ret:
            if k not in render_outs:
                render_outs[k] = []
            render_outs[k].append(ret[k])

        render_outs = {k : torch.cat(render_outs[k], 0) for k in render_outs}
        
        return render_outs
    
    def forward(self, data:dict, back_net=None):
        
        back_img = None

        if cfg.back_net.back_net_on and not cfg.use_back_img_gt:
            
            with torch.no_grad():
                
                if cfg.db != 'humman':
                    back_net_input = torch.cat([data['inp_img_normed'],data['inp_heatmap'][None]], dim=1)
                else:
                    back_net_input = torch.cat([data['inp_squared'], data['inp_heatmap'][None]], dim=1)
                
                back_img = back_net(back_net_input)

                if cfg.db == 'humman':
                    back_img = back_img[:,:,140:500]
                
                back_img = torch.flip(back_img, dims=[-1]) # flip back to original view
                data.update({'back_img': back_img})
        
        if cfg.use_back_img_gt:
            back_img = data['back_img_gt']
            data.update({'back_img': back_img})
        
        # encode images
        if cfg.back_net.back_net_on or cfg.use_back_img_gt:
            img_enc_input = torch.cat([data['inp_img_normed'], back_img], dim=0)
        else:
            img_enc_input = data['inp_img_normed']
        
        img_enc_out = self.img_encoder(img_enc_input)
        
        data.update({'inp_fmap':img_enc_out['fmap'][0:1]})
        if cfg.back_net.back_net_on or cfg.use_back_img_gt:
            data.update({'back_fmap':img_enc_out['fmap'][1:2]})
        
        # Setting rays and sample points along the rays
        rays_o, rays_d = data['rays']
        rays_o = torch.reshape(rays_o, [-1,3])
        rays_d = torch.reshape(rays_d, [-1,3])
        
        # motion info setting
        motion_info = self.humannerf.motion_infos_forward(**data)
        data.update(motion_info)
        
        render_outs = self.process_humannerf(data, rays_o, rays_d, data['near'], data['far'])
        if cfg.back_net.back_net_on or cfg.use_back_img_gt:
            render_outs.update({'back_img': data['back_img']})
        
        return render_outs
