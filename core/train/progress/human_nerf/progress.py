import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import skimage
from core.data import create_dataloader
from core.utils.network_util import set_requires_grad
from third_parties.lpips import LPIPS
from core.utils.train_util import cpu_data_to_gpu, remove_batch_axis
from core.utils.image_util import tile_images, to_8b_image, output_to_image, to_8b3ch_image
from core.utils.eval_utils import create_evaluate_result_save_file, update_evaluate_result_save_file, get_lpips_metric, psnr_metric
from core.utils.log_util import Logger
from configs import cfg

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']
np.set_printoptions(precision=5, suppress=True)
class Progress_Renderer(object):
    def __init__(self, fast_validation=False, test_mode=False):
        self.device = None
        self.prog_dataloader = create_dataloader('progress', fast_validation=False)
        self.val_dataloader = create_dataloader('validation', fast_validation=fast_validation and (not test_mode))

        test_protocol = 'novel_pose_test' if cfg.novel_pose_test else 'novel_view_test'

        self.logdir = logdir = cfg.logdir
        if test_mode:
            test_result_dir_name = 'test_results_cliff_smpl' if cfg.cliff_estimated_smpl_test else 'test_results'
            if cfg.diff_angle_test:
                self.test_result_dir = os.path.join(logdir, test_result_dir_name, 'diff_angle_test')
            else:
                self.test_result_dir = os.path.join(logdir, test_result_dir_name, test_protocol)
                
            validation_result_dir = self.test_result_dir
        
        else:
            validation_result_dir = os.path.join(logdir, 'val', test_protocol)
        
        if not cfg.diff_angle_test:
            self.val_logger = Logger(os.path.join(validation_result_dir, 'val_results.txt'), clean_up=False, create_yaml=False)
        
        self.val_outdir = validation_result_dir

        self.progress_outdir = logdir if not test_mode else os.path.join(self.test_result_dir)
        if not test_mode:
            self.sample_rendering_outdir = os.path.join(logdir, 'val', test_protocol, 'vis_results') 
        else: 
            self.sample_rendering_outdir = os.path.join(self.test_result_dir, 'vis_results')
        
        if not os.path.exists(self.val_outdir):
            os.makedirs(self.val_outdir)
        if test_mode:
            if not os.path.exists(self.test_result_dir):
                os.makedirs(self.test_result_dir)
        
        create_evaluate_result_save_file(self.val_outdir)
        self.lpips = None

    def set_device(self, device):
        self.device = device
        
    def progress_begin(self, network):
        network.eval()
        cfg.perturb = 0.

    def progress_end(self, network):
        network.train()
        cfg.perturb = cfg.train.perturb
    
    def render_progress(self, network, iter, back_net=None):
        self.progress_begin(network)
        print('Evaluate Progress Images ...')
        images = []
        is_empty_img = False
        dataloader = self.prog_dataloader

        for _, batch in enumerate(tqdm(dataloader)):
            # only access the first batch as we process one image one time
            batch = remove_batch_axis(batch)
            
            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']
            
            rendered = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
            alpha_render = np.full(
                        (height * width), np.array([0])/255., 
                        dtype='float32')
            batch['iter_val'] = torch.full((1,), iter)
            data = cpu_data_to_gpu(batch, device=self.device, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            
            
            with torch.no_grad():
                net_output = network(data, back_net)

            rgb = net_output['rgb'].cpu().numpy()
            alpha = net_output['alpha'].cpu().numpy()
            target_rgbs = batch['target_rgbs']
            
            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs
            alpha_render[ray_mask] = alpha
            
            truth = to_8b_image(truth.reshape((height, width, -1)))
            rendered = to_8b_image(rendered.reshape((height, width, -1)))
            alpha_render = to_8b3ch_image(alpha_render.reshape((height, width, -1)))
            input_img = to_8b_image(data['inp_img'].data.cpu().numpy().reshape((height, width, -1)))
            
            sample_img = [input_img]
            
            if cfg.back_net.back_net_on:
                back_gt = to_8b_image(batch['back_img_gt'][0].permute(1,2,0).cpu().numpy())
                back_pred = to_8b_image(net_output['back_img'][0].permute(1,2,0).cpu().numpy())
                sample_img.extend([back_gt, back_pred])
            
            if cfg.use_uv_inpainter:
                uv_map_pred = net_output['uv_map_pred']
                uv_map_pred = to_8b_image(uv_map_pred[0].permute(1,2,0).cpu().numpy())
                uv_map_pred = np.array(Image.fromarray(uv_map_pred).resize((width, height), Image.BILINEAR), dtype=np.uint8)
                uv_map_gt = to_8b_image(data['uv_map_gt'].data.cpu().numpy())
                uv_map_gt = np.array(Image.fromarray(uv_map_gt).resize((width, height), Image.BILINEAR), dtype=np.uint8)
                sample_img.extend([uv_map_gt, uv_map_pred])
            
            sample_img.extend([alpha_render, rendered, truth])
            sample_img = np.concatenate(sample_img, axis=1)
            
            images.append(sample_img)
            
            # check if we create empty images (only at the begining of training)
            if iter <= 5000 and \
                np.allclose(rendered, np.array(cfg.bgcolor), atol=5.):
                is_empty_img = True
                break
            
        if not is_empty_img:
            tiled_image = tile_images(images)
            if not os.path.exists(self.progress_outdir):
                os.makedirs(self.progress_outdir)
            Image.fromarray(tiled_image).save(
                os.path.join(self.progress_outdir, "prog_{:06}.jpg".format(iter)))
        
        self.progress_end(network)

        return is_empty_img
            
    def validate(self, network, iter, writer=None, save_samples=True, back_net=None):
        self.progress_begin(network)
        if self.lpips is None:
            print('LPIPS module(VGG) is not defined, newly create LPIPS instance.')
            self.lpips = LPIPS(net='vgg').to(self.device)
            set_requires_grad(self.lpips, requires_grad=False)
            
        print('Validation ...')

        psnr_l = []
        ssim_l = []
        lpips_l = []
        if cfg.diff_angle_test:
            performance_rec_obs_view_wise = {}
        dataloader = self.val_dataloader

        total_inference_time = 0.
        for bidx, batch in enumerate(tqdm(dataloader)):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # only access the first batch as we process one image one time
            batch = remove_batch_axis(batch)
            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            rendered = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
            alpha_render = np.full(
                        (height * width), np.array([0])/255., 
                        dtype='float32')
            batch['iter_val'] = torch.full((1,), iter)
            data = cpu_data_to_gpu(batch, device=self.device, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            
            start_event.record()
            with torch.no_grad():
                net_output = network(data, back_net)
            end_event.record()
            torch.cuda.synchronize()
            inference_time = start_event.elapsed_time(end_event) / 1000. # convert to seconds
            total_inference_time += inference_time
            
            rgb = net_output['rgb'].data.cpu().numpy()
            alpha = net_output['alpha'].data.cpu().numpy()
            target_rgbs = batch['target_rgbs']
            
            pred_img, _, gt_img = output_to_image(width, height, ray_mask, np.array(cfg.bgcolor)/255., rgb, alpha, target_rgbs)

            pred_img_norm, gt_img_norm = pred_img / 255., gt_img / 255.
            
            ray_mask_np = ray_mask.reshape(height, width).cpu().numpy().astype(np.uint8)
            x, y, w, h = cv2.boundingRect(ray_mask_np)
            sample_psnr = psnr_metric(pred_img_norm[y:y+h, x:x+w], gt_img_norm[y:y+h, x:x+w])
            
            image_pred = pred_img_norm[y:y+h, x:x+w]
            image_gt = gt_img_norm[y:y+h, x:x+w]
            sample_ssim = skimage.metrics.structural_similarity(image_pred, image_gt, multichannel=True)
            sample_lpips_loss = get_lpips_metric(lpips=self.lpips, rgb=torch.from_numpy(image_pred).float().unsqueeze(0).to('cuda'), 
                                                 target=torch.from_numpy(image_gt).float().unsqueeze(0).to('cuda'))
            
            psnr_l.append(sample_psnr)
            ssim_l.append(sample_ssim)
            lpips_l.append(sample_lpips_loss)

            if cfg.diff_angle_test:
                if batch['obs_view_index'].item() not in performance_rec_obs_view_wise:
                    performance_rec_obs_view_wise[batch['obs_view_index'].item()] = {}
                if batch['subject'][0] not in performance_rec_obs_view_wise:
                    performance_rec_obs_view_wise[batch['obs_view_index'].item()][batch['subject'][0]] = {}
                    performance_rec_obs_view_wise[batch['obs_view_index'].item()][batch['subject'][0]].update({'psnr':[], 'ssim':[], 'lpips':[]})
                performance_rec_obs_view_wise[batch['obs_view_index'].item()][batch['subject'][0]]['psnr'].append(sample_psnr)
                performance_rec_obs_view_wise[batch['obs_view_index'].item()][batch['subject'][0]]['ssim'].append(sample_ssim)
                performance_rec_obs_view_wise[batch['obs_view_index'].item()][batch['subject'][0]]['lpips'].append(sample_lpips_loss)
            
            if save_samples:
                sample_save_path = os.path.join(self.sample_rendering_outdir, 'iter_{}'.format(iter))

                if not os.path.exists(sample_save_path):
                    os.makedirs(sample_save_path)
                rendered[ray_mask] = rgb
                truth[ray_mask] = target_rgbs
                alpha_render[ray_mask] = alpha

                truth = to_8b_image(truth.reshape((height, width, -1)))
                rendered = to_8b_image(rendered.reshape((height, width, -1)))
                alpha_render = to_8b3ch_image(alpha_render.reshape((height, width, -1)))
                input_img = to_8b_image(data['inp_img'].data.cpu().numpy().reshape((height, width, -1)))

                sample_img = [input_img]
                if cfg.back_net.back_net_on:
                    back_gt = to_8b_image(batch['back_img_gt'][0].permute(1,2,0).cpu().numpy())
                    back_pred = to_8b_image(net_output['back_img'][0].permute(1,2,0).cpu().numpy())
                    sample_img.extend([back_gt, back_pred])
                
                if cfg.use_uv_inpainter:
                    uv_map_pred = net_output['uv_map_pred']
                    uv_map_pred = to_8b_image(uv_map_pred[0].permute(1,2,0).cpu().numpy())
                    uv_map_pred = np.array(Image.fromarray(uv_map_pred).resize((width, height), Image.BILINEAR), dtype=np.uint8)
                    uv_map_gt = to_8b_image(data['uv_map_gt'].data.cpu().numpy())
                    uv_map_gt = np.array(Image.fromarray(uv_map_gt).resize((width, height), Image.BILINEAR), dtype=np.uint8)
                    sample_img.extend([uv_map_gt, uv_map_pred])
                
                sample_img.extend([alpha_render, rendered, truth])
                sample_img = np.concatenate(sample_img, axis=1)
                
                filename = 'S{}_{}_obsview_{:04d}_tgtview_{:04d}_'.format(batch['subject'][0], 
                                                                          batch['frame_name'][0], 
                                                                          batch['obs_view_index'].item(), 
                                                                          batch['view_index'].item())
                filename += 'obspose_{:02d}_tgtpose_{:02d}_'.format(batch['obs_pose_index'].item(), 
                                                                    batch['pose_index'].item())
                filename += 'PSNR_{:.2f}_LPIPS_{:.2f}.png'.format(sample_psnr, sample_lpips_loss*1000.)
                
                Image.fromarray(sample_img).save(os.path.join(sample_save_path, filename))

        average_inference_time = total_inference_time / len(dataloader)
        
        if not cfg.diff_angle_test:
            psnr = np.array(psnr_l).mean()
            ssim = np.array(ssim_l).mean()
            lpips = np.array(lpips_l).mean()*1000

            new_results = {'psnr':psnr, 'ssim':ssim, 'lpips': lpips}

            update_evaluate_result_save_file(new_results, iter, self.val_outdir)

            if writer is not None:
                writer.add_scalar('val/psnr', psnr, iter)
                writer.add_scalar('val/ssim', ssim, iter)
                writer.add_scalar('val/lpips', lpips, iter)
            
            print('#### Validation Results ####')
            print('Task {} | Current iter {} \n'.format(
                    'Test', iter))
            print('PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}\n'.format(
                    psnr, ssim ,lpips))
            self.val_logger.write('Task {} | Current iter {} \n'.format(
                    'Test', iter))
            self.val_logger.write('PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}\n'.format(
                    psnr, ssim, lpips))
        else:
            print("#### Test Results on Various Observation Views ####")
            for obs_view in performance_rec_obs_view_wise:
                subjects_psnr_list = []
                subjects_ssim_list = []
                subjects_lpip_list = []
                for subject in performance_rec_obs_view_wise[obs_view]:
                    subject_psnr = np.array(performance_rec_obs_view_wise[obs_view][subject]['psnr']).mean()
                    subject_ssim = np.array(performance_rec_obs_view_wise[obs_view][subject]['ssim']).mean()
                    subject_lpips = np.array(performance_rec_obs_view_wise[obs_view][subject]['lpips']).mean()*1000

                    subjects_psnr_list.append(subject_psnr)
                    subjects_ssim_list.append(subject_ssim)
                    subjects_lpip_list.append(subject_lpips)
                
                cur_obs_view_psnr = np.array(subjects_psnr_list).mean()
                cur_obs_view_ssim = np.array(subjects_ssim_list).mean()
                cur_obs_view_lpips = np.array(subjects_lpip_list).mean()

                with open(os.path.join(self.val_outdir, 'diff_angle_test_results.txt'.format(obs_view)), 'a') as f:
                    f.write('Observed View {} | PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}\n'.format(obs_view, 
                                                                                               cur_obs_view_psnr, 
                                                                                               cur_obs_view_ssim, 
                                                                                               cur_obs_view_lpips))
                    f.close()
                print('Observed View {} | PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}'.format(obs_view, cur_obs_view_psnr, cur_obs_view_ssim, cur_obs_view_lpips))
        print(f"Average inference time: {average_inference_time:.4f} seconds")