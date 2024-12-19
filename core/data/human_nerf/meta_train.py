
import os
import pickle

import numpy as np
import cv2
import torch
import torch.utils.data

from core.utils.image_util import load_image
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes, query_dst_skeleton
from core.utils.file_util import list_files, split_path
from core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox, sample_patch_rays

from configs import cfg
import pdb

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            mode,
            dataset_path,
            target_view='0',
            maxframes=-1,
            keyfilter=None,
            bgcolor=None,
            ray_shoot_mode='image',
            **_):

        print('[Dataset Path]', dataset_path) 
        print('[Mode]', mode)

        self.mode = mode

        self.dataset_path = dataset_path
        
        self.tasks = [str(x) for x in cfg.meta_dataset_cfg[cfg.db].all_tasks]
        
        self.num_sample_meta_train = {'zju_mocap':10, 'aist':10, 'fashionvideo':10}
        self.view = target_view
        
        self.maxframes = maxframes
        self.num_frames = 10 # dummy value
        self.keyfilter = keyfilter
        self.bgcolor = bgcolor

        self.db = self.load_db()
        
        self.ray_shoot_mode = ray_shoot_mode

    def load_db(self): # DB key elems : self.view
        db = {k: {} for k in self.tasks}
        for subject in self.tasks:
            canonical_joints, canonical_bbox = \
                self.load_canonical_joints(subject=subject, view=self.view)
            if 'motion_weights_priors' in self.keyfilter:
                motion_weights_priors = \
                    approx_gaussian_bone_volumes(
                        canonical_joints,
                        canonical_bbox['min_xyz'],
                        canonical_bbox['max_xyz'],
                        grid_size=cfg.mweight_volume.volume_size).astype('float32') # (25,32,32,32)

            cameras = self.load_train_cameras(subject=subject, view=self.view)
            mesh_infos = self.load_train_mesh_infos(subject=subject, view=self.view)

            framelist = self.load_train_frames(subject=subject, view=self.view)
            framelist = np.array(framelist)
            
            print(f' -- Subject {subject} Total Frames: {len(framelist)}')

            db[subject]['canonical_joints'] = canonical_joints.copy()
            db[subject]['canonical_bbox'] = canonical_bbox.copy()
            db[subject]['motion_weights_priors'] = motion_weights_priors.copy()
            db[subject]['cameras'] = cameras.copy()
            db[subject]['mesh_infos'] = mesh_infos.copy()
            db[subject]['framelist'] = framelist.copy()

            # select training samples
            indices = np.arange(len(framelist))

            n_train = self.num_sample_meta_train[cfg.db]

            train_ratio = np.floor(len(framelist)/n_train).astype(int)
            train_indices = indices[::train_ratio][:n_train]
            train_mask = np.zeros((len(framelist))).astype(bool)
            train_mask[train_indices] = True
            train_framelist = framelist[train_mask]
            db[subject]['train_framelist'] = train_framelist.copy()

            # select eval samples
            remain_mask = ~train_mask
            test_framelist = framelist[remain_mask]
            db[subject]['test_framelist'] = test_framelist.copy()
            db[subject]['val_framelist'] = test_framelist.copy()[::cfg.train.val_subsample]
            
        return db.copy()
        
    def load_canonical_joints(self, subject, view):
        cl_joint_path = os.path.join(self.dataset_path, subject, view, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        return canonical_joints, canonical_bbox

    def load_train_cameras(self, subject, view):
        cameras = None
        with open(os.path.join(self.dataset_path, subject, view, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return cameras

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self, subject, view):
        mesh_infos = None
        if subject == 'A10guZxkDfS':
            if 'non_relative' in cfg.experiment:
                with open(os.path.join(self.dataset_path, subject, view, 'mesh_infos_non_relative.pkl'), 'rb') as f:   
                    mesh_infos = pickle.load(f)
            else:
                with open(os.path.join(self.dataset_path, subject, view, 'mesh_infos.pkl'), 'rb') as f:   
                    mesh_infos = pickle.load(f)
        else:
            with open(os.path.join(self.dataset_path, subject, view, 'mesh_infos.pkl'), 'rb') as f:   
                mesh_infos = pickle.load(f)

        for frame_name in mesh_infos.keys():
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox

        return mesh_infos

    def load_train_frames(self, subject, view):
        img_paths = list_files(os.path.join(self.dataset_path, subject, view, 'images'),
                               exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]
    
    def load_image(self, frame_name:str, subject:str, view:str, cameras, bg_color):
        imagepath = os.path.join(self.dataset_path, subject, view, 'images', '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.dataset_path, subject, view,
                                'masks', 
                                '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))
        
        # undistort image
        if frame_name in cameras and 'distortions' in cameras[frame_name]:
            K = cameras[frame_name]['intrinsics']
            D = cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask

    def __len__(self):
        return self.num_frames
    
    def set_task(self, task:str=None):
        
        self.subject = np.random.choice(np.array([x for x in self.db.keys()])) if task == None else task 
        
        self.framelist_total = self.db[self.subject]['framelist'].copy()
        
        if self.mode == 'train':
            self.framelist = self.db[self.subject]['train_framelist'].copy()
        elif self.mode == 'validation':
            self.framelist = self.db[self.subject]['val_framelist'].copy()

        # Set data when inference mode is progress mode
        elif self.mode == 'progress':
            self.framelist = self.db[self.subject]['val_framelist'].copy()
            if self.maxframes > 0:
                self.framelist = self.framelist[:self.maxframes]
        elif self.mode == 'test':
            self.framelist = self.db[self.subject]['test_framelist'].copy()
        
        else: # for movement rendering, tpose, freeview
            self.framelist = self.db[self.subject]['framelist'].copy()

        self.num_frames = len(self.framelist)
        
    def __getitem__(self, idx): # get single image's data component
        
        subject = self.subject
        
        framelist = self.framelist
        frame_name = framelist[idx]
        cameras = self.db[subject]['cameras']

        results = {
            'frame_name': frame_name
        }

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        img, alpha = self.load_image(frame_name, subject, self.view, cameras, bgcolor)
        img = (img / 255.).astype('float32')

        results.update({'img': img,
                        'alpha': alpha,
                        })

        H, W = img.shape[0:2]

        dst_skel_info = query_dst_skeleton(self.db[subject]['mesh_infos'], frame_name)
        dst_bbox = dst_skel_info['bbox'] # dict, contains {minxyz, maxxyz} got from 3D skeleton
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        
        assert frame_name in cameras
        K = cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2] *= cfg.resize_img_scale

        E_data = cameras[frame_name]['extrinsics']
        E = apply_global_tfm_to_camera(
                E=E_data, 
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]
        
        results.update({'dst_bbox_min': dst_bbox['min_xyz'],
                        'dst_bbox_max': dst_bbox['max_xyz'],
                        'E_st': E_data,
                        'K': K,
                        'Rh':dst_skel_info['Rh'],
                        'Th':dst_skel_info['Th'],
                        })
                        
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        ray_img = img.reshape(-1, 3) 
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')

        if self.ray_shoot_mode == 'image':
            pass
        elif self.ray_shoot_mode == 'patch':
            rays_o, rays_d, ray_img, near, far, \
            target_patches, patch_masks, patch_div_indices, mask_raw = \
                sample_patch_rays(cfg=cfg, img=img, H=H, W=W,
                                       subject_mask=alpha[:, :, 0] > 0.,
                                       bbox_mask=ray_mask.reshape(H, W),
                                       ray_mask=ray_mask,
                                       rays_o=rays_o, 
                                       rays_d=rays_d, 
                                       ray_img=ray_img, 
                                       near=near, 
                                       far=far)
        else:
            assert False, f"Ivalid Ray Shoot Mode: {self.ray_shoot_mode}"
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 
        
        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor})

            if self.ray_shoot_mode == 'patch':
                results.update({
                    'patch_div_indices': patch_div_indices,
                    'patch_masks': patch_masks,
                    'target_patches': target_patches})

        if 'target_rgbs' in self.keyfilter:
            results['target_rgbs'] = ray_img

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints
                )
            cnl_gtfms = get_canonical_global_tfms(
                            self.db[subject]['canonical_joints'])
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms
            })

        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = self.db[subject]['motion_weights_priors'].copy()

        # get the bounding box of canonical volume
        if 'cnl_bbox' in self.keyfilter:
            canonical_bbox = self.db[self.subject]['canonical_bbox']
            min_xyz = canonical_bbox['min_xyz'].astype('float32')
            max_xyz = canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
            })
        
        return results