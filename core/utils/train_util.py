import time
import torch
import numpy as np
import torch.nn as nn
from core.utils.ssim import ssim
from third_parties.lpips import LPIPS
## data and loss handling Functions

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)
bce_loss = nn.BCELoss()

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class Loss(nn.Module):
    def __init__(self, device, cfg):
        super(Loss, self).__init__()
        self.cfg = cfg
        self.device = device
        self.lossweights = cfg.train.lossweights
        self.loss_names = cfg.train.lossweights.keys()
        
        if "lpips" in self.loss_names:
            self.lpips = LPIPS(net='vgg')
            set_requires_grad(self.lpips, requires_grad=False)
            self.lpips = self.lpips.to(device)
        
        if cfg.use_uv_inpainter:
            if cfg.uv_map.uv_l1_loss:
                self.lossweights['uv_l1'] = 1.0
            if cfg.uv_map.uv_lpips_loss:
                self.lossweights['uv_lpips'] = 1.0    
    
    def forward(self, lossweights:dict, net_output:dict, 
                patch_masks:torch.Tensor, 
                bgcolor:torch.Tensor, 
                target_rgbs:torch.Tensor,
                target_alpha:torch.Tensor,
                div_indices:torch.Tensor, 
                uv_map_gt=None,
                **kwargs):
        
        loss_names = list(lossweights.keys())
        
        rgb = net_output['rgb']
        alpha = net_output['alpha']
        
        ray_mask = kwargs['ray_mask']

        if target_alpha.dtype == torch.bool:
            target_alpha = target_alpha.float()

        if not self.cfg.train.ray_shoot_mode == 'patch':
            height = kwargs['image_height']
            width = kwargs['image_width']
            tight_ray_mask_param = kwargs['tight_ray_mask']
            rgb_full = bgcolor.broadcast_to(height * width, 3).clone().float()
            target_rgbs_full = bgcolor.broadcast_to(height * width, 3).clone().float()
            
            rgb_full[ray_mask] = rgb
            target_rgbs_full[ray_mask] = target_rgbs
            target_alpha_full = target_alpha
            rgb_full = rgb_full.view(height, width, 3).unsqueeze(0)
            target_rgbs_full = target_rgbs_full.view(height, width, 3).unsqueeze(0)
            
            losses = get_img_rebuild_loss(self.lpips, loss_names, rgb_full, target_rgbs_full, alpha=alpha, target_alpha=target_alpha_full, ray_mask=ray_mask, tight_ray_mask_param=tight_ray_mask_param)
        
        else:
            height = None
            width = None
            tight_ray_mask_param = None
            rgb_patch = unpack_patch_imgs(rgb, patch_masks, bgcolor, target_rgbs, div_indices)
            target_rgbs_patch = target_rgbs
            pred_alpha_patch = unpack_patch_masks(alpha, patch_masks, div_indices)
            target_alpha_patch = target_alpha
            losses = get_img_rebuild_loss(self.lpips, loss_names, rgb_patch, target_rgbs_patch, alpha=pred_alpha_patch, target_alpha=target_alpha_patch)
        
        if self.cfg.use_uv_inpainter:
            uv_map_gt = uv_map_gt[None].permute(0,3,1,2)
            uv_map_pred = net_output['uv_map_pred']
            
            if self.cfg.uv_map.uv_l1_loss:
                losses['uv_l1'] = img2l1(uv_map_pred, uv_map_gt)

            if self.cfg.uv_map.uv_lpips_loss:
                losses['uv_lpips'] = torch.mean(self.lpips(scale_for_lpips(uv_map_pred), scale_for_lpips(uv_map_gt)))

        # Exclude losses in loss_names but not defined in losses
        train_losses = []
        for k,v in losses.items():
            if k not in ['adv_D', 'tex_adv_D']:
                train_losses.append(lossweights[k]*v)
        item_values = {k: lossweights[k]*v.item() for k,v in losses.items()}
        
        return sum(train_losses), item_values
    
def get_img_rebuild_loss(lpips, loss_names, rgb, target_rgb, alpha=None, target_alpha=None, ray_mask=None, tight_ray_mask_param=None):
    losses = {}
    if tight_ray_mask_param is not None:
        x = tight_ray_mask_param[0].item()
        y = tight_ray_mask_param[1].item()
        w = tight_ray_mask_param[2].item()
        h = tight_ray_mask_param[3].item()
    
    if "mse" in loss_names:
        losses["mse"] = img2mse(rgb, target_rgb)
    
    if "ssim" in loss_names:
        losses["ssim"] = torch.mean(1-ssim(rgb.permute(0,3,1,2), target_rgb.permute(0,3,1,2), data_range=1, size_average=False))
        
    if "l1" in loss_names:
        losses["l1"] = img2l1(rgb, target_rgb)
    
    if "lpips" in loss_names:
        if tight_ray_mask_param is not None:
            rgb = rgb[:,y:y+h, x:x+w, :]
            target_rgb = target_rgb[:,y:y+h, x:x+w, :]
        lpips_loss = lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                                scale_for_lpips(target_rgb.permute(0, 3, 1, 2)))
        losses["lpips"] = torch.mean(lpips_loss)

    if "mask" in loss_names:
        losses["mask"] = img2mse(alpha, target_alpha)

    return losses

def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.

def unpack_patch_imgs(rgbs, patch_masks, bgcolor, targets, div_indices)->torch.Tensor:
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch
    patch_imgs = bgcolor.expand(targets.shape).clone() # mask square shaped patche images
    
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]
    
    return patch_imgs

def unpack_patch_masks(alpha, patch_masks, div_indices)->torch.Tensor:
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    patch_imgs = torch.zeros(patch_masks.shape).float().to(patch_masks.device)
    
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = alpha[div_indices[i]:div_indices[i+1]]
    
    return patch_imgs

def unpack_depths(depths, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    patch_imgs = bgcolor.expand(targets.shape[:-1]).clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = depths[div_indices[i]:div_indices[i+1]]

    return patch_imgs

def calc_depth_reg_loss(depths):
    N_patch, size, size = depths.shape
    for i in range(size-1):
        for j in range(size-1):
            diff = 0
            diff += torch.abs((depths[:,i,j]-depths[:,i+1,j]))
            diff += torch.abs((depths[:,i,j]-depths[:,i,j+1]))
    loss = torch.sum(diff)

    return loss

## Misc Functions
def numpy_to_torch(np_data, exclude_keys=['frame_name', 'img_width', 'img_height']):
    if exclude_keys is None:
        exclude_keys = []

    torch_data = {}
    for key,val in np_data.items():
        if key in exclude_keys:
            torch_data[key] = val
        else:
            if isinstance(val, dict):
                torch_data[key] = {x:torch.from_numpy(y) for x,y in val.items() if isinstance(val, np.ndarray)}
            elif isinstance(val, np.ndarray):
                torch_data[key] = torch.from_numpy(val)
            elif isinstance(val, list):
                torch_data[key] = torch.tensor(val)
            else:
                torch_data[key] = val
    
    return torch_data

def torch_to_numpy(torch_data, exclude_keys=['frame_name', 'img_width', 'img_height']):
    if exclude_keys is None:
        exclude_keys = []

    np_data = {}
    for key,val in torch_data.items():
        if key in exclude_keys:
            np_data[key] = val
        else:
            if isinstance(val, dict):
                np_data[key] = {x:y.detach().cpu().numpy() for x,y in val.items() if isinstance(val, torch.Tensor)}
            elif isinstance(val, torch.Tensor):
                np_data[key] = val.detach().cpu().numpy()
            elif isinstance(val, list):
                np_data[key] = np.array([v.detach().cpu().numpy() for v in val])
            else:
                np_data[key] = val
    
    return np_data

def list_dict_data(data:dict):
    print("All items")
    print('\n'.join([str((k, type(v))) for k,v in data.items()]))
    print("Tensor items")
    print('\n'.join([str((k, type(v), v.shape, v.dtype)) for k,v in data.items() if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)]))
    print("List, Tuple items")
    print('\n'.join([str((k, type(v), len(v))) for k,v in data.items() if type(v) in [list, tuple]]))
    print("String items")
    print('\n'.join([str((k, type(v), v)) for k,v in data.items() if isinstance(v, str)]))
    print("Dict items")
    print('\n'.join([str((k, type(v), v.keys())) for k,v in data.items() if isinstance(v, dict)]))
    print("Sub Dict items\n")
    
    for k,v in data.items():
        if type(v) == dict:
            subdict = v
            print(f"Sub Dict of {k}")
            list_dict_data(subdict)

def remove_batch_axis(data, exclude_keys=None)->dict:
    # This method removes batch axis from data, number of batch should be 1.
    for key, val in data.items():
        if exclude_keys is not None and key in exclude_keys:
            continue

        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if exclude_keys is not None and key in exclude_keys:
                    continue
                if isinstance(sub_val, torch.Tensor) or isinstance(sub_val, np.ndarray):
                    assert sub_val.shape[0] == 1
                    val[sub_key] = sub_val[0]
        elif isinstance(val, torch.Tensor) or isinstance(val, np.ndarray):
            assert val.shape[0] == 1
            data[key] = val[0]
    return data

def gpu_data_to_cpu(gpu_data, exclude_keys=None, detach=False):
    def to_cpu(data, detach=False):
        if detach:
            data = data.detach().cpu()
        else:
            data = data.cpu()
        return data
    
    if exclude_keys is None:
        exclude_keys = []

    cpu_data = {}
    for key, val in gpu_data.items():
        if key in exclude_keys:
            continue

        if isinstance(val, list):
            assert len(val) > 0
            if not isinstance(val[0], str): # ignore string instance
                cpu_data[key] = [to_cpu(x, detach) for x in val]
        elif isinstance(val, dict):
            cpu_data[key] = {sub_k: to_cpu(sub_val, detach) for sub_k, sub_val in val.items()}
        else:
            cpu_data[key] = to_cpu(val, detach)

    return cpu_data

def cpu_data_to_gpu(cpu_data, device, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = []

    gpu_data = {}
    for key, val in cpu_data.items():
        if key in exclude_keys:
            continue

        if isinstance(val, list):
            assert len(val) > 0
            if not isinstance(val[0], str): # ignore string instance
                gpu_data[key] = [x.to(device) for x in val]
        elif isinstance(val, dict):
            gpu_data[key] = {sub_k: sub_val.to(device) for sub_k, sub_val in val.items()}
        elif isinstance(val, torch.Tensor):
            gpu_data[key] = val.to(device)
        else:
            gpu_data[key] = val

    return gpu_data

## Timer
class Timer():
    def __init__(self):
        self.curr_time = 0

    def begin(self):
        self.curr_time = time.time()

    def log(self):
        diff_time = time.time() - self.curr_time
        self.begin()
        
        return diff_time # in seconds
        