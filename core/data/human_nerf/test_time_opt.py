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
    approx_gaussian_bone_volumes
from core.utils.file_util import list_files, split_path
from core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox

from configs import cfg
import pdb

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            mode,
            data_type,
            infer_mode,
            dataset_path,
            target_view='0',
            maxframes=-1,
            ntrf=None,
            adapt=False,
            keyfilter=None,
            bgcolor=None,
            ray_shoot_mode='image',
            skip=1,
            all_subjects=[],
            **_):

        print('[Dataset Path]', dataset_path) 
        print('[Mode]', mode)
        print('[Data Type]', data_type)
        print('[Infernece Mode]', infer_mode)

        self.mode = mode
        self.adapt = adapt
        self.data_type = data_type
        self.infer_mode = infer_mode
        self.dataset_path = dataset_path
        
        self.tasks = all_subjects
        
        self.view = target_view
        self.ntrf = ntrf
        self.maxframes = maxframes
        self.num_frames = 10 # dummy value
        self.skip = skip
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
            
            print(f' -- Subject {subject} Total Frames: {len(framelist)}')

            db[subject]['canonical_joints'] = canonical_joints.copy()
            db[subject]['canonical_bbox'] = canonical_bbox.copy()
            db[subject]['motion_weights_priors'] = motion_weights_priors.copy()
            db[subject]['cameras'] = cameras.copy()
            db[subject]['mesh_infos'] = mesh_infos.copy()
            db[subject]['framelist'] = np.array(framelist).copy()
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
    
    def query_dst_skeleton(self, mesh_infos, frame_name):
        return {
            'poses': mesh_infos[frame_name]['poses'].astype('float32'),
            'dst_tpose_joints': \
                mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'bbox': mesh_infos[frame_name]['bbox'].copy(),
            'Rh': mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': mesh_infos[frame_name]['Th'].astype('float32')
        }

    @staticmethod
    def select_rays(select_inds, rays_o, rays_d, ray_img, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        ray_img = ray_img[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, ray_img, near, far
    
    def get_patch_ray_indices(
            self, 
            N_patch, 
            ray_mask, 
            subject_mask, 
            bbox_mask,
            patch_size, 
            H, W):

        assert subject_mask.dtype == np.bool
        assert bbox_mask.dtype == np.bool

        bbox_exclude_subject_mask = np.bitwise_and(
            bbox_mask,
            np.bitwise_not(subject_mask)
        )
        list_ray_indices = []
        list_mask = []
        list_xy_min = []
        list_xy_max = []

        total_rays = 0
        patch_div_indices = [total_rays]
        for i in range(N_patch):
            # let p = cfg.patch.sample_subject_ratio
            # prob p: we sample on subject area
            # prob (1-p): we sample on non-subject area but still in bbox
            if np.random.rand(1)[0] < cfg.patch.sample_subject_ratio:
                candidate_mask = subject_mask
            else:
                candidate_mask = bbox_exclude_subject_mask

            ray_indices, mask, xy_min, xy_max = \
                self._get_patch_ray_indices(ray_mask, candidate_mask, 
                                            patch_size, H, W, i)

            assert len(ray_indices.shape) == 1
            total_rays += len(ray_indices)
            list_ray_indices.append(ray_indices)
            list_mask.append(mask)
            list_xy_min.append(xy_min)
            list_xy_max.append(xy_max)
            
            patch_div_indices.append(total_rays)

        select_inds = np.concatenate(list_ray_indices, axis=0)
        patch_info = {
            'mask': np.stack(list_mask, axis=0),
            'xy_min': np.stack(list_xy_min, axis=0),
            'xy_max': np.stack(list_xy_max, axis=0)
        }
        patch_div_indices = np.array(patch_div_indices)

        return select_inds, patch_info, patch_div_indices


    def _get_patch_ray_indices(
            self, 
            ray_mask, 
            candidate_mask, 
            patch_size, 
            H, W, patch_count=0):

        assert len(ray_mask.shape) == 1
        assert ray_mask.dtype == np.bool
        assert candidate_mask.dtype == np.bool

        valid_ys, valid_xs = np.where(candidate_mask)

        # determine patch center
        select_idx = np.random.choice(valid_ys.shape[0], 
                                      size=[1], replace=False)[0]
        center_x = valid_xs[select_idx]
        center_y = valid_ys[select_idx]

        # determine patch boundary
        half_patch_size = patch_size // 2
        x_min = np.clip(a=center_x-half_patch_size, 
                        a_min=0, 
                        a_max=W-patch_size)
        x_max = x_min + patch_size
        y_min = np.clip(a=center_y-half_patch_size,
                        a_min=0,
                        a_max=H-patch_size)
        y_max = y_min + patch_size

        sel_ray_mask = np.zeros_like(candidate_mask)
        sel_ray_mask[y_min:y_max, x_min:x_max] = True

        #####################################################
        ## Below we determine the selected ray indices
        ## and patch valid mask

        sel_ray_mask = sel_ray_mask.reshape(-1)
        inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
        select_masked_inds = np.where(inter_mask)

        masked_indices = np.cumsum(ray_mask) - 1
        select_inds = masked_indices[select_masked_inds]
        
        inter_mask = inter_mask.reshape(H, W)
        # cv2.imwrite('patch_mask_{}.jpg'.format(patch_count),(inter_mask[...,None]*255.).astype(np.uint8))
        return select_inds, \
                inter_mask[y_min:y_max, x_min:x_max], \
                np.array([x_min, y_min]), np.array([x_max, y_max])
    
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

    def sample_patch_rays(self, img, H, W,
                          subject_mask, bbox_mask, ray_mask,
                          rays_o, rays_d, ray_img, near, far):

        select_inds, patch_info, patch_div_indices = \
            self.get_patch_ray_indices(
                N_patch=cfg.patch.N_patches, 
                ray_mask=ray_mask, 
                subject_mask=subject_mask, 
                bbox_mask=bbox_mask,
                patch_size=cfg.patch.size, 
                H=H, W=W)

        rays_o, rays_d, ray_img, near, far = self.select_rays(
            select_inds, rays_o, rays_d, ray_img, near, far)
        
        targets = []
        for i in range(cfg.patch.N_patches):
            x_min, y_min = patch_info['xy_min'][i] 
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(img[y_min:y_max, x_min:x_max])
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)

        patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

        return rays_o, rays_d, ray_img, near, far, \
                target_patches, patch_masks, patch_div_indices

    def __len__(self):
        return self.num_frames
    
    def set_task(self, task:str=None):
        self.subject  = np.random.choice(self.db.keys()) if task == None else task
        self.framelist_total = self.db[self.subject]['framelist'].copy()
        
        # Set data according to data type
        if 'unseen' in self.data_type:
            if self.ntrf < 0:
                raise ValueError('For unseen scenario, number of training frames should be set bigger than 0')
            
            if self.ntrf == 10 and cfg.fixed_frame_set_experiment:
                if cfg.subject == 392:
                    frame_train = [0,70,150,225,275,315,360,387,471,530]
                    index_mask = np.zeros(len(self.framelist_total))
                    index_mask[frame_train] = 1
                    train_data_mask = index_mask.astype(bool)

                elif cfg.subject == 393:
                    frame_train = [0,75,113,145,240,323,362,421,453,560]
                    index_mask = np.zeros(len(self.framelist_total))
                    index_mask[frame_train] = 1
                    train_data_mask = index_mask.astype(bool)
            
            else:
                # Set training frame set from first of frame sequence, test frames are remainings
                frame_train = np.arange(self.ntrf)
                index_mask = np.zeros(len(self.framelist_total))
                index_mask[frame_train] = 1
                train_data_mask = index_mask.astype(bool)
            
            test_data_mask = ~train_data_mask
            if self.mode == 'train':
                self.framelist = self.framelist_total[train_data_mask]
            if self.mode == 'test':
                self.framelist = self.framelist_total[test_data_mask] if self.data_type != 'unseen_recon' else self.framelist_total[train_data_mask]

        elif self.data_type == 'recon':
            train_data_mask = np.ones(len(self.framelist_total)).astype(bool)
            test_data_mask = train_data_mask
            self.framelist = self.framelist_total[train_data_mask]
        
        # Set data when inference mode is progress mode
        if self.infer_mode == 'progress':
            self.framelist = self.framelist_total[test_data_mask] if self.data_type != 'unseen_recon' else self.framelist_total[train_data_mask]
            self.framelist = self.framelist[::self.skip]
            if self.maxframes > 0:
                self.framelist = self.framelist[:self.maxframes]
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

        dst_skel_info = self.query_dst_skeleton(self.db[subject]['mesh_infos'], frame_name)
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
            target_patches, patch_masks, patch_div_indices = \
                self.sample_patch_rays(img=img, H=H, W=W,
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