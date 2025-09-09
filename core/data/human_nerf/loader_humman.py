import os
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data
from core.utils.camera_util import construct_ray_data, apply_global_tfm_to_camera
from configs import cfg
import random
import torchvision.transforms as transforms
from PIL import Image
import pdb

class Opt():
    def __init__(self):
        # Thuman2 operation options
        self.transform = ['resize']
        self.resize = 640
        self.mean = 0
        self.std = 1

class BaseTransform:
    def __init__(self, opt):
        self.opt = opt
        self.functions = []
        self.collect()

    def collect(self):
        raise NotImplementedError

    def __call__(self, input_tensor):
        return self.functions(input_tensor)
    
class ImageTransform(BaseTransform):
    def __init__(self, opt):
        super(ImageTransform, self).__init__(opt)

    def __call__(self, input_tensor):
        if isinstance(input_tensor, Image.Image): input_tensor = input_tensor.convert('RGB')
        return super().__call__(input_tensor)

    def collect(self):
        if 'resize' in self.opt.transform:
            self.functions.append(transforms.Resize((self.opt.resize, self.opt.resize),
                                                    interpolation=transforms.InterpolationMode.NEAREST))
        self.functions.append(transforms.ToTensor())
        if 'normalize' in self.opt.transform:
            self.functions.append(transforms.Normalize(self.opt.mean, self.opt.std, inplace=True))

        self.functions = transforms.Compose(self.functions)

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, mode,
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            bgcolor=None,
            ray_shoot_mode='image',
            target_view='0',
            fast_validation=True,
            novel_pose_test=False,
            **_):
        
        self.mode = mode
        train_subject_path = 'subject_list/HuMMan_train_list.txt'
        val_subject_path = 'subject_list/HuMMan_test_list.txt'
        subjects_path = train_subject_path if self.mode == 'train' else val_subject_path
        self.subjects = self.load_subjects(subjects_path)
        self.dataset_path = 'dataset/humman'
        self.opt = Opt()
        self.image_transform = ImageTransform(self.opt)
        print('[Dataset Path]', self.dataset_path)
        self.training_views = np.arange(10)
        self.test_obs_views = [0,4,8]
        self.pose_interval = 1
        self.novel_view_pose_start = 0
        self.novel_pose_pose_start = 0
        self.image_scaling = 1/3
        
        self.pose_start = 0
        self.num_pose = 17

        if self.mode == 'train':
            self.mesh_infos_all, self.cameras_all, self.smpl_visibility_all = self.load_dataset()
            self.num_data_samples = len(self.subjects) * len(self.training_views) * self.num_pose

        elif self.mode == 'progress':
            (self.mesh_infos_all, 
             self.cameras_all, 
             self.smpl_visibility_all, 
             self.obs_view_list, 
             self.target_view_list, 
             self.obs_frame_path_list,
             self.target_frame_path_list,
             self.obs_pose_list, 
             self.target_pose_list, 
             self.subject_idx_list, 
             self.subject_names) = self.load_test_dataset(novel_pose_test, False)
            
            indexes = np.array([661, 1171, 1726, 1797, 2150, 
                                       2280, 2298, 2359, 3563, 3976, 
                                       4049, 4145, 4330, 4534, 4971, 
                                       5076, 5712, 5840, 5970, 6091, 
                                       6706, 6960, 7213, 7287, 7573, 
                                       7821, 8183, 8299, 8605, 8683, 
                                       9211, 9252]) # 32 fixed sample
            if maxframes > 0:
                self.obs_view_list = self.obs_view_list[indexes]
                self.target_view_list = self.target_view_list[indexes]
                self.obs_frame_path_list = self.obs_frame_path_list[indexes]
                self.target_frame_path_list = self.target_frame_path_list[indexes]
                self.obs_pose_list = self.obs_pose_list[indexes]
                self.target_pose_list = self.target_pose_list[indexes]
                self.subject_idx_list = self.subject_idx_list[indexes]
                self.subject_names = self.subject_names[indexes]
            self.num_data_samples = len(indexes)

        elif self.mode == 'validation':
            (self.mesh_infos_all, 
             self.cameras_all, 
             self.smpl_visibility_all, 
             self.obs_view_list, 
             self.target_view_list, 
             self.obs_frame_path_list,
             self.target_frame_path_list,
             self.obs_pose_list, 
             self.target_pose_list, 
             self.subject_idx_list, 
             self.subject_names) = self.load_test_dataset(novel_pose_test, fast_validation)
            
            self.num_data_samples = len(self.target_view_list)
        
        self.keyfilter = keyfilter
        self.bgcolor = bgcolor

        self.ray_shoot_mode = ray_shoot_mode
        if mode == 'train':
            print(f' -- Total Training Frames: {self.num_data_samples}')
        elif mode == 'progress':
            print(f' -- Total Progress Frames: {self.num_data_samples}')
        elif mode == 'validation':
            print(f' -- Total Validation Frames: {self.num_data_samples}')
        
    def load_subjects(self, path):
        file = open(path)
        lines = file.readlines()
        items = []
        for line in lines:
            item = line.strip('\n')
            items.append(item)
        
        return np.array(items)
            
    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }
    
    @staticmethod
    def load_image(dataset_path, subject, keyword, view_index, frame_name, ext='jpg') -> Image.Image:
        if keyword == 'img':
            ext = 'png'
        elif keyword == 'mask':
            ext = 'png'

        imagepath = os.path.join(dataset_path, subject, f'{keyword}', f'camera{view_index:06d}', f'{frame_name}'+f'.{ext}')
        image = Image.open(imagepath)

        return image

    @staticmethod
    def mask_image(image: Image.Image, mask: Image.Image, bgcolor : np.ndarray) -> Image.Image:
        w, h = image.size
        black_canvas = Image.new('RGB', (w, h))
        bgcolor_canvas = Image.new('RGB', (w, h), color=tuple(bgcolor.astype(np.uint8).tolist()))
        image_blk = Image.composite(image, black_canvas, mask)
        image_bgcolor = Image.composite(image, bgcolor_canvas, mask)
        
        return image_blk, image_bgcolor

    @staticmethod
    def generate_2d_heatmap(joints, image_res):
        """
        :param joints: 2D joints (24, 2)
        :param image_res: (w, h)
        :return: 2D heatmap (24, h, w)
        """
        w, h = image_res
        joints = joints[:, :2]
        heatmap = np.zeros((joints.shape[0], h, w))
        
        sigma = 6
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        gussian_kernel = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        for idx, pt in enumerate(joints):
            x, y = int(pt[0]), int(pt[1])
            if x<0 or x>=w or y<0 or y>=h:
                continue

            ul = np.round(x - 3 * sigma -1).astype(int), np.round(y - 3 * sigma -1).astype(int) # upper left, y_u, x_l
            br = np.round(x + 3 * sigma +2).astype(int), np.round(y + 3 * sigma +2).astype(int) # bottom right, y_b, x_r
            
            c,d = max(0, -ul[0]), min(br[0], w) - ul[0] # c: y_u, d: y_b
            a,b = max(0, -ul[1]), min(br[1], h) - ul[1] # a: x_l, b: x_r

            cc,dd = max(0, ul[0]), min(br[0], w) # cc: y_u, dd: y_b
            aa,bb = max(0, ul[1]), min(br[1], h) # aa: x_l, bb: x_r

            heatmap[idx, aa:bb, cc:dd]= np.maximum(heatmap[idx, aa:bb, cc:dd], gussian_kernel[a:b, c:d])

        return heatmap
    
    def load_2d_pose(self, joints, E, K):
        R = E[:3, :3]
        T = E[:3, 3]

        joints_world = joints
        joints_cam = joints_world @ R.T + T
        joints_2d = joints_cam @ K.T
        joints_2d = joints_2d[:, :2] / joints_2d[:, 2:]

        return joints_2d

    def pad_to_square(self, image):
        # Image : PIL Image or np.array

        image_np = np.array(image)

        width, height = image_np.shape[1], image_np.shape[0]
        new_size = max(width, height)
        
        # Calculate padding
        pad_width = (new_size - width) // 2
        pad_height = (new_size - height) // 2
        
        # Pad the image
        padded_image = np.pad(image_np, 
                              ((pad_height, new_size - height - pad_height),
                               (pad_width, new_size - width - pad_width),
                               (0, 0)),
                              mode='constant', constant_values=0)
        
        return padded_image
    
    def load_dataset(self):
        dataset_dir = self.dataset_path
        subjects = self.subjects
        
        mesh_infos_all = []
        cameras_all = []
        smpl_vis_mask_all = []
        
        for idx, subj_name in enumerate(subjects):
            # load frame name, camera data, mesh data, canonical joint data
            subj_dir = os.path.join(dataset_dir, f'{subj_name}')
            
            with open(os.path.join(subj_dir, 'mesh_infos.pkl'), 'rb') as f:
                mesh_infos = pickle.load(f)
            with open(os.path.join(subj_dir, 'cameras.pkl'), 'rb') as f:
                cameras = pickle.load(f)
            with open(os.path.join(subj_dir, 'smpl_vertex_visibility_mask.pkl'), 'rb') as f:
                smpl_vis_mask = pickle.load(f)
            
            mesh_infos_all.append(mesh_infos)
            cameras_all.append(cameras)
            smpl_vis_mask_all.append(smpl_vis_mask)

        return mesh_infos_all, cameras_all, smpl_vis_mask_all

    def load_test_dataset(self, novel_pose_test=False, fast_validation=True):
        dataset_dir = self.dataset_path
        subjects = self.subjects
        
        mesh_infos_all = []
        cameras_all = []
        smpl_vis_mask_all = []
        
        subject_idx_list = []
        obs_view_list = []
        obs_pose_list = []
        target_view_list = []
        target_pose_list = []
        subject_names = []
        obs_frame_path_list = []
        target_frame_path_list = []
        
        for idx, subj_name in enumerate(subjects):
            # load frame name, camera data, mesh data, canonical joint data
            subj_dir = os.path.join(dataset_dir, subj_name)
            
            with open(os.path.join(subj_dir, 'mesh_infos.pkl'), 'rb') as f:
                mesh_infos = pickle.load(f)
            with open(os.path.join(subj_dir, 'cameras.pkl'), 'rb') as f:
                cameras = pickle.load(f)
            with open(os.path.join(subj_dir, 'smpl_vertex_visibility_mask.pkl'), 'rb') as f:
                smpl_vis_mask = pickle.load(f)

            mesh_infos_all.append(mesh_infos)
            cameras_all.append(cameras)
            smpl_vis_mask_all.append(smpl_vis_mask)

            target_view_indexes = np.arange(10)
            
            for ovidx, obs_view_index in enumerate(self.test_obs_views):
                for target_view in target_view_indexes:
                    if not novel_pose_test: # For novel view synthesis test, exclude same view
                        if target_view == obs_view_index:
                            continue
                    obs_view_dir = os.path.join(subj_dir, 'kinect_color', f'kinect_{obs_view_index:03d}')
                    target_view_dir = os.path.join(subj_dir, 'kinect_color', f'kinect_{target_view:03d}')
                    
                    frames = sorted(os.listdir(target_view_dir))
                    frames = frames[:self.num_pose] if len(frames) > self.num_pose else frames
                    
                    if novel_pose_test:
                        target_frames = frames[1:].copy()
                    if not novel_pose_test:
                        target_frames = frames.copy()
                    
                    for pose_idx, file_name in enumerate(target_frames):
                        target_frame_path = os.path.join(target_view_dir, file_name)
                        if not novel_pose_test:
                            obs_pose_idx = pose_idx
                            observation_frame_path = os.path.join(obs_view_dir, file_name)
                        else:
                            obs_pose_idx = self.pose_start
                            observation_frame_path = os.path.join(obs_view_dir, frames[self.pose_start])
                        
                        obs_view_list.append(obs_view_index)
                        target_view_list.append(target_view)
                        obs_frame_path_list.append(observation_frame_path)
                        target_frame_path_list.append(target_frame_path)
                        obs_pose_list.append(obs_pose_idx)
                        target_pose_list.append(pose_idx)
                        subject_idx_list.append(idx)
                        subject_names.append(subj_name)
        
        obs_view_list = np.array(obs_view_list)
        target_view_list = np.array(target_view_list)
        obs_frame_path_list = np.array(obs_frame_path_list)
        target_frame_path_list = np.array(target_frame_path_list)
        obs_pose_list = np.array(obs_pose_list)
        target_pose_list = np.array(target_pose_list)
        subject_idx_list = np.array(subject_idx_list)
        subject_names = np.array(subject_names)
        
        if fast_validation:
            obs_view_list = obs_view_list[::40]
            target_view_list = target_view_list[::40]
            obs_pose_list = obs_pose_list[::40]
            target_pose_list = target_pose_list[::40]
            subject_idx_list = subject_idx_list[::40]
            subject_names = subject_names[::40]

        return (mesh_infos_all,
                cameras_all, 
                smpl_vis_mask_all, 
                obs_view_list, 
                target_view_list, 
                obs_frame_path_list, 
                target_frame_path_list, 
                obs_pose_list, 
                target_pose_list, 
                subject_idx_list, 
                subject_names)
    
    def __len__(self):
        return self.num_data_samples

    def __getitem__(self, idx): # get single image's data component
        
        # Prepare data index
        if self.mode == 'train':
            subject_idx = idx // (self.num_pose * len(self.training_views))
            subject_name = self.subjects[subject_idx]

            view_idx = idx % len(self.training_views)
            obs_view_index = np.random.choice(self.training_views)
            back_view_index = obs_view_index

            pose_start = self.pose_start
            pose_idx = (idx % (len(self.training_views) * self.num_pose)) // len(self.training_views) * self.pose_interval + pose_start
            all_pose_idxes = np.arange(0, 17)
            obs_pose_idx = pose_idx if not cfg.random_pose_training else np.random.choice(all_pose_idxes)
            back_pose_index = obs_pose_idx

            frame_num = pose_idx * 6
            frame_name = f'{frame_num:06d}'

            obs_frame_num = obs_pose_idx * 6
            obs_frame_name = f'{obs_frame_num:06d}'
            
            target_frame_path = os.path.join(self.dataset_path, subject_name, 'kinect_color', f'kinect_{view_idx:03d}', f'{frame_name}.png')
            target_mask_path = os.path.join(self.dataset_path, subject_name, 'kinect_mask', f'kinect_{view_idx:03d}', f'{frame_name}.png')
            obs_frame_path = os.path.join(self.dataset_path, subject_name, 'kinect_color', f'kinect_{obs_view_index:03d}', f'{obs_frame_name}.png')
            obs_mask_path = os.path.join(self.dataset_path, subject_name, 'kinect_mask', f'kinect_{obs_view_index:03d}', f'{obs_frame_name}.png')
            back_img_frame_path = os.path.join(self.dataset_path, subject_name, 'back_renderings', 'kinect_color', f'kinect_{obs_view_index:03d}', f'{obs_frame_name}.png')
            
            if not os.path.exists(back_img_frame_path):
                available_back_imgs = os.listdir(os.path.join(self.dataset_path, subject_name, 'back_renderings', 'kinect_color', f'kinect_{view_idx:03d}'))
                
                available_back_img = random.choice(available_back_imgs)
                # Update pose index and frame information
                obs_frame_name = available_back_img.split('.')[0]
                obs_frame_num = int(obs_frame_name)
                obs_pose_idx = int(obs_frame_num // 6) 
                
                frame_name = obs_frame_name
                frame_num = obs_frame_num
                pose_idx = obs_pose_idx

                obs_frame_path = os.path.join(self.dataset_path, subject_name, 'kinect_color', f'kinect_{obs_view_index:03d}', f'{obs_frame_name}.png')
                obs_mask_path = os.path.join(self.dataset_path, subject_name, 'kinect_mask', f'kinect_{obs_view_index:03d}', f'{obs_frame_name}.png')
                back_img_frame_path = os.path.join(self.dataset_path, subject_name, 'back_renderings', 'kinect_color', f'kinect_{obs_view_index:03d}', f'{obs_frame_name}.png')
                target_frame_path = os.path.join(self.dataset_path, subject_name, 'kinect_color', f'kinect_{view_idx:03d}', f'{frame_name}.png')
                target_mask_path = os.path.join(self.dataset_path, subject_name, 'kinect_mask', f'kinect_{view_idx:03d}', f'{frame_name}.png')

        else:
            subject_idx = self.subject_idx_list[idx]
            subject_name = self.subject_names[idx]
            
            view_idx = self.target_view_list[idx]
            obs_view_index = self.obs_view_list[idx]
            back_view_index = obs_view_index
            
            pose_start = self.pose_start
            pose_idx = self.target_pose_list[idx]
            obs_pose_idx = self.obs_pose_list[idx]
            back_pose_index = obs_pose_idx

            obs_frame_path = self.obs_frame_path_list[idx]
            target_frame_path = self.target_frame_path_list[idx]
            frame_name = target_frame_path.split('/')[-1].split('.')[0]
            target_mask_path = os.path.join(self.dataset_path, subject_name, 'kinect_mask', f'kinect_{view_idx:03d}', f'{frame_name}.png')
            
            obs_frame_name = obs_frame_path.split('/')[-1].split('.')[0]
            obs_mask_path = os.path.join(self.dataset_path, subject_name, 'kinect_mask', f'kinect_{obs_view_index:03d}', f'{obs_frame_name}.png')
            
            back_img_frame_path = os.path.join(self.dataset_path, subject_name, 'back_renderings', 'kinect_color', f'kinect_{obs_view_index:03d}', f'{obs_frame_name}.png')
        
        results = { 'frame_name': f'{frame_name}',
                    'subject': subject_name,
                    'view_index': view_idx,
                    'pose_index': pose_idx,
                    'obs_view_index': obs_view_index,
                    'obs_pose_index': obs_pose_idx,
                    'back_view_index': back_view_index,
                    'back_pose_index': back_pose_index,
                    }
        
        bgcolor = np.array([0,0,0], dtype='float32')

        mesh_infos = self.mesh_infos_all[subject_idx]
        cameras = self.cameras_all[subject_idx]
        smpl_visibilities = self.smpl_visibility_all[subject_idx]

        camera = cameras[f'{view_idx:06d}']
        mesh_info = mesh_infos[f'{frame_name}']
        subject_dir = os.path.join(self.dataset_path, subject_name)
        image_path = target_frame_path
        mask_path = target_mask_path
        
        resize_H, resize_W = 360, 640

        img_pil = Image.open(image_path).resize((resize_W, resize_H))
        alpha_pil = Image.open(mask_path).resize((resize_W, resize_H))
        _, img_pil = self.mask_image(img_pil, alpha_pil, bgcolor)
        img = np.array(img_pil).astype(np.float32)/255.
        alpha = np.array(alpha_pil.convert('RGB')).astype('float32') / 255.
        
        if cfg.use_uv_inpainter:
            uvmap_gt_path = os.path.join(subject_dir, 'uvmap_gt', 'uvmap_gt.png')
            uv_map_gt = np.array(Image.open(uvmap_gt_path).convert('RGB'))
            uv_map_gt = cv2.resize(uv_map_gt, (cfg.uv_map.uv_map_size, cfg.uv_map.uv_map_size), interpolation=cv2.INTER_LINEAR)
            uv_map_gt = uv_map_gt.astype(np.float32) / 255.
            results.update({'uv_map_gt': uv_map_gt})
        H, W = np.array(img_pil).shape[0:2] 

        tgt_bbox = self.skeleton_to_bbox(mesh_info['joints'])
        tgt_betas = mesh_info['betas'].astype(np.float32)
        tgt_poses = mesh_info['poses'].astype(np.float32)
        tgt_tpose_joints = mesh_info['tpose_joints'].astype(np.float32)
        tgt_Rh = mesh_info['Rh'].astype(np.float32)
        tgt_Th = mesh_info['Th'].astype(np.float32)

        dst_skel_info = {'betas': tgt_betas, 'poses': tgt_poses, 'dst_tpose_joints': tgt_tpose_joints, 'Rh': tgt_Rh, 'Th': tgt_Th, 'bbox': tgt_bbox}

        info = {'img': np.array(img_pil).astype(np.float32)/255.,
                'alpha': alpha,
                'alpha_1ch': np.array(alpha_pil).astype(np.float32)/255.,
                'W': W,
                'H': H,
                'skel_info':dst_skel_info,
                'camera_dict':camera,
                'resize_img_scale':1/3,
                'ray_mode':self.ray_shoot_mode,
                'keyfilter':self.keyfilter}
        
        if self.ray_shoot_mode == 'patch':
            info.update({'N_patches':cfg.patch.N_patches,
                         'sample_subject_ratio':cfg.patch.sample_subject_ratio,
                         'patch_size':cfg.patch.size})
        target_ray_data = construct_ray_data(**info)
        
        # Target pose data
        results.update({'img':img,
                        'alpha': alpha,
                        'img_width': W,
                        'img_height': H,
                        'bgcolor': bgcolor
                        })
        
        results.update({
                        'target_poses_69': tgt_poses[3:],
                        'target_betas': tgt_betas
                        })
        
        results.update(target_ray_data)
        
        # Preprocess of input image's data
        obs_camera = cameras[f'{obs_view_index:06d}']
        obs_img_pil = Image.open(obs_frame_path).resize((resize_W, resize_H))
        obs_alpha_pil = Image.open(obs_mask_path).resize((resize_W, resize_H))
        obs_alpha = (np.array(obs_alpha_pil.convert('RGB')).astype('float32') / 255.)[...,[0]]
        
        _, obs_img_pil = self.mask_image(obs_img_pil, obs_alpha_pil, bgcolor)
        
        obs_squared = self.pad_to_square(obs_img_pil)
        obs_img_normed = (np.array(obs_img_pil).astype(np.float32)/255.).transpose(2,0,1)[None]
        obs_squared_normed = self.image_transform(Image.fromarray(obs_squared))[None]
        
        obs_E_data = obs_camera['extrinsics'].astype('float32')
        obs_K = obs_camera['intrinsics'].astype('float32')
        obs_K[:2] *= cfg.resize_img_scale

        obs_mesh_info = mesh_infos[obs_frame_name]
        obs_Rh = obs_mesh_info['Rh']
        obs_Th = obs_mesh_info['Th']

        obs_E = apply_global_tfm_to_camera(obs_E_data, obs_Rh, obs_Th).astype('float32')
        
        obs_betas = obs_mesh_info['betas'].astype('float32')
        obs_poses = obs_mesh_info['poses'].astype('float32')
        obs_joints = obs_mesh_info['joints'].astype('float32')
        
        obs_visible_vertices_mask = smpl_visibilities[f'{obs_view_index:06d}'][obs_frame_name].astype(np.float32)
        
        obs_bbox = self.skeleton_to_bbox(obs_mesh_info['joints'])
        obs_bbox_min_xyz = obs_bbox['min_xyz'].astype('float32')
        obs_bbox_max_xyz = obs_bbox['max_xyz'].astype('float32')
        obs_bbox_scale_xyz = (2 / (obs_bbox_max_xyz - obs_bbox_min_xyz)).astype('float32')
        
        obs_H, obs_W = np.array(obs_img_pil).shape[0:2]
        obs_R = obs_E[:3, :3]
        obs_T = obs_E[:3, 3]
        obs_ray_o = -np.dot(obs_R.T, obs_T).ravel()

        # make 2d pose heatmap
        obs_joints_2d = self.load_2d_pose(obs_joints, obs_E, obs_K)
        obs_joints_2d[:,1] = obs_joints_2d[:,1] + 140 # 140 is the offset value in square image
        hmap_size = max(obs_H, obs_W)
        obs_joints_2d_heatmap = self.generate_2d_heatmap(obs_joints_2d, (hmap_size, hmap_size)).astype(np.float32)
        heatmap_input = obs_joints_2d_heatmap
        

        # Observation data
        results.update({'inp_img':np.array(obs_img_pil).astype(np.float32)/255.,
                        'inp_squared':obs_squared_normed,
                        'inp_alpha': obs_alpha,
                        'inp_img_width': W,
                        'inp_img_height': H,
                        'inp_extrinsics':obs_E,
                        'inp_intrinsics':obs_K,
                        'inp_img_normed':obs_img_normed,
                        'inp_heatmap':heatmap_input,
                        'inp_ray_o':obs_ray_o,
                        'inp_betas':obs_betas,
                        'inp_visible_vertices_mask':obs_visible_vertices_mask,
                        'inp_bbox_min_xyz': obs_bbox_min_xyz,
                        'inp_bbox_scale_xyz': obs_bbox_scale_xyz,
                        'inp_poses_69': obs_poses[3:]
                        })
        
        # Preprocess of back image's data
        if cfg.back_net.back_net_on:
            back_frame_name = obs_frame_name
            back_img_pil = Image.open(back_img_frame_path).resize((resize_W, resize_H))
            back_alpha_pil = obs_alpha_pil.transpose(Image.FLIP_LEFT_RIGHT)
            back_img_black_pil, back_img_pil = self.mask_image(back_img_pil, back_alpha_pil, bgcolor)
            back_squared = self.pad_to_square(back_img_pil)
            back_squared_normed = self.image_transform(Image.fromarray(back_squared))[None]
            back_img_normed = (np.array(back_img_pil, dtype=np.float32)/255.).transpose(2,0,1)[None]
            back_camera = cameras[f'{back_view_index:06d}']
            back_E_data = back_camera['extrinsics'].astype('float32')
            back_K = back_camera['intrinsics'].astype('float32')
            back_K[:2] *= cfg.resize_img_scale

            back_Rh = mesh_infos[back_frame_name]['Rh']
            back_Th = mesh_infos[back_frame_name]['Th']
            
            back_E = apply_global_tfm_to_camera(back_E_data, back_Rh, back_Th).astype('float32')

            results.update({'back_extrinsics':back_E,
                            'back_intrinsics':back_K,
                            'back_img_gt':back_img_normed,
                            })

        
        ############################## Visualization ##############################
        # vis_hmap = heatmap_input.copy()
        # vis_hmap = heatmap_to_jet(vis_hmap)
        # vis_hmap = opencv_image_blending(obs_squared, vis_hmap, 0.5)
        # vis = np.concatenate([obs_squared, back_squared, vis_hmap], axis=1)
        # os.makedirs('vis_humman_test', exist_ok=True)
        # Image.fromarray(vis).save(f'vis_humman_test/{idx:06d}.png')
        # print(f'{idx+1}/{self.num_data_samples} saved')
        ############################################################################
        
        return results
    
if __name__ == '__main__':
    dataset = Dataset('validation', dataset_path='dataset/humman', novel_pose_test=True, fast_validation=False)
    for sampleidx in range(len(dataset)):
        sample = dataset[sampleidx]