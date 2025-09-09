import os
import joblib
import torch
import numpy as np

def psnr_metric(img_pred, img_gt):
    mse = np.mean((img_pred - img_gt)**2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr

def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.

def lpips_metric(lpips, rgb, target):
    lpips_loss = lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                       scale_for_lpips(target.permute(0, 3, 1, 2)))
    return torch.mean(lpips_loss).cpu().detach().numpy()

def get_lpips_metric(lpips, rgb, target):
    lpips_loss = lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                       scale_for_lpips(target.permute(0, 3, 1, 2)))
    return torch.mean(lpips_loss).cpu().detach().numpy()

def create_evaluate_result_save_file(directory_path, file_name='data.pt'):
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    file_path = os.path.join(directory_path, file_name)
    
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        init_data = {'psnr':{}, 'ssim':{}, 'lpips':{}} # init data
        
        joblib.dump(init_data, file_path)

        return init_data

def update_evaluate_result_save_file(new_data:dict, iter:int, directory_path:str, file_name:str='data.pt'):
    
    file_path = os.path.join(directory_path, file_name)
    
    # check if file exists
    if os.path.exists(file_path):
        data = joblib.load(file_path)
    else:
        data = {'psnr':{}, 'ssim':{}, 'lpips':{}}
    
    data['psnr'].update({str(iter): new_data['psnr']})
    data['ssim'].update({str(iter): new_data['ssim']})
    data['lpips'].update({str(iter): new_data['lpips']})

    joblib.dump(data, file_path)

    return data