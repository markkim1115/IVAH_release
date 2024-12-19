import pdb
import os
import pickle

import numpy as np
import cv2
import torch
import torch.utils.data
from core.utils.body_util import body_pose_to_body_RTs, approx_gaussian_bone_volumes
from core.utils.camera_util import construct_ray_data, apply_global_tfm_to_camera
from configs import cfg
import torchvision.transforms as transforms
from PIL import Image

class Opt():
    def __init__(self):
        # Thuman2 operation options
        self.transform = ['resize']
        self.resize = 512
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
        train_subject_path = 'subject_list/thuman2_datalist_train.txt'
        val_subject_path = 'subject_list/thuman2_datalist_test.txt'
        subjects_path = train_subject_path if self.mode == 'train' else val_subject_path
        if self.mode != 'train':
            self.test_obs_views = cfg.progress.test_obs_view_list
        self.subjects = self.load_subjects(subjects_path)
        self.dataset_path = 'dataset/thuman2'
        self.image_transform = ImageTransform(Opt())
        print('[Dataset Path]', self.dataset_path)
        self.canonical_joints, self.canonical_bbox = None, None
        
        if self.mode == 'train':
            self.datalist = self.load_dataset()
        elif self.mode == 'progress':
            self.datalist = self.load_test_dataset(novel_pose_test, False)
        elif self.mode == 'validation':
            diff_angle_test = cfg.diff_angle_test
            self.datalist = self.load_test_dataset(novel_pose_test, fast_validation, diff_angle_test)
        
        if mode == 'progress':
            skip = 10
            self.datalist = np.array(self.datalist)
            indexes = np.array([13, 25, 57, 81, 137, 188, 362, 436, 479, 
                                647, 671, 690, 710, 726, 801, 844, 890, 
                                922, 974, 995, 1015, 1045, 1103, 1132, 
                                1160, 1250, 1300, 1350, 1398, 1516, 1559, 1604]) # 30 samples
            if maxframes > 0:
                # self.datalist = self.datalist[:maxframes]
                self.datalist = self.datalist[indexes]
        
        self.keyfilter = keyfilter
        self.bgcolor = bgcolor

        self.ray_shoot_mode = ray_shoot_mode
        if mode == 'train':
            print(f' -- Total Training Frames: {self.get_total_frames()}')
        elif mode == 'progress':
            print(f' -- Total Progress Frames: {self.get_total_frames()}')
        elif mode == 'validation':
            print(f' -- Total Validation Frames: {self.get_total_frames()}')

    def load_subjects(self, path):
        file = open(path)
        lines = file.readlines()
        items = []
        for line in lines:
            item = line.strip('\n').split(' ')[0]
            items.append(int(item))
        
        return np.array(sorted(items))
            
    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - 0.3
        max_xyz = np.max(skeleton, axis=0) + 0.3

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }
    
    @staticmethod
    def load_image(dataset_path, subject, keyword, frame_name) -> Image.Image:
        imagepath = os.path.join(dataset_path, subject, f'{keyword}', '{}.png'.format(frame_name))
        image = Image.open(imagepath)

        return image

    @staticmethod
    def mask_image(image: Image.Image, mask: Image.Image, bgcolor : np.ndarray) -> Image.Image:
        h, w = image.size
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

    # def load_image(self, subject, frame_name, camera, bg_color):
    #     imagepath = os.path.join(self.dataset_path, subject, 'images', f'{frame_name}.png')
    #     orig_img = np.array(load_image(imagepath))

    #     maskpath = os.path.join(self.dataset_path, subject, 'masks', '{}.png'.format(frame_name))
    #     alpha_mask = np.array(load_image(maskpath))
        
    #     # undistort image
    #     if 'distortions' in camera:
    #         K = camera[frame_name]['intrinsics']
    #         D = camera[frame_name]['distortions']
    #         orig_img = cv2.undistort(orig_img, K, D)
    #         alpha_mask = cv2.undistort(alpha_mask, K, D)

    #     alpha_mask = alpha_mask / 255.
    #     img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]

    #     black_bg = np.array([0,0,0]).astype(np.float32)

    #     black_bg_img = alpha_mask * orig_img + (1.0 - alpha_mask) * black_bg[None, None, :]
    #     if cfg.resize_img_scale != 1.:
    #         img = cv2.resize(img, None, 
    #                             fx=cfg.resize_img_scale,
    #                             fy=cfg.resize_img_scale,
    #                             interpolation=cv2.INTER_LANCZOS4)
    #         alpha_mask = cv2.resize(alpha_mask, None, 
    #                                 fx=cfg.resize_img_scale,
    #                                 fy=cfg.resize_img_scale,
    #                                 interpolation=cv2.INTER_LINEAR)
    #         black_bg_img = cv2.resize(black_bg_img, None, 
    #                                 fx=cfg.resize_img_scale,
    #                                 fy=cfg.resize_img_scale,
    #                                 interpolation=cv2.INTER_LINEAR)
                                
    #     return img, alpha_mask

    def load_dataset(self):
        dataset_dir = self.dataset_path
        subjects = self.subjects
        data_samples = []
        
        for idx, subj_name in enumerate(subjects):
            # load frame name, camera data, mesh data, canonical joint data
            subj_dir = os.path.join(dataset_dir, f'{subj_name:04d}')
            
            with open(os.path.join(subj_dir, 'mesh_infos.pkl'), 'rb') as f:
                mesh_infos = pickle.load(f)
            with open(os.path.join(subj_dir, 'cameras.pkl'), 'rb') as f:
                cameras = pickle.load(f)
            with open(os.path.join(subj_dir, 'canonical_joints.pkl'), 'rb') as f:
                canonical_joints = pickle.load(f)
            
            with open(os.path.join(subj_dir, 'smpl_vertex_visibility_mask.pkl'), 'rb') as f:
                smpl_vis_mask = pickle.load(f)
            if idx == 0:
                # This data will be used as common data for all subjects
                self.canonical_joints = canonical_joints['joints']
                self.canonical_bbox = self.skeleton_to_bbox(canonical_joints['joints'])

            framenames = np.array(list(mesh_infos.keys()))
            framenames = framenames[::3] if self.mode == 'train' else framenames[::10]
            
            view_indexes = np.arange(360)

            for jdx, frame_name in enumerate(framenames):
                target_view_angle = int(frame_name)
                view_index = view_indexes[target_view_angle]
                data_sample = {}
                data_sample['db'] = cfg.db
                data_sample['subject'] = f'{subj_name:04d}'
                data_sample['view_index'] = view_index
                data_sample['frame_name'] = frame_name # saved without extension
                data_sample['pose_index'] = 0
                mesh_info = mesh_infos[frame_name]
                bbox = self.skeleton_to_bbox(mesh_info['joints'])
                
                dst_skel_info = {}
                dst_skel_info['betas'] = mesh_info['betas'].astype('float32')
                dst_skel_info['poses'] = mesh_info['poses'].astype('float32')
                dst_skel_info['dst_tpose_joints'] = mesh_info['tpose_joints'].astype('float32')
                dst_skel_info['Rh'] = mesh_info['Rh'].astype('float32')
                dst_skel_info['Th'] = mesh_info['Th'].astype('float32')
                dst_skel_info['bbox'] = bbox # (min_xyz, max_xyz)

                data_sample['dst_skel_info'] = dst_skel_info
                data_sample['camera'] = cameras[frame_name]

                # observation setting
                inp_basename = np.random.choice(framenames, replace=False)
                
                data_sample['inp_framename'] = inp_basename
                data_sample['inp_maskname'] = inp_basename
                data_sample['obs_view_index'] = int(inp_basename)
                data_sample['obs_pose_index'] = 0
                data_sample['inp_Rh'] = mesh_infos[inp_basename]['Rh'].astype(np.float32)
                data_sample['inp_Th'] = mesh_infos[inp_basename]['Th'].astype(np.float32)
                data_sample['inp_betas'] = mesh_infos[inp_basename]['betas'].astype(np.float32)
                data_sample['inp_poses'] = mesh_infos[inp_basename]['poses'].astype(np.float32)
                data_sample['inp_joints'] = mesh_infos[inp_basename]['joints'].astype(np.float32)
                data_sample['inp_tpose_joints'] = mesh_infos[inp_basename]['tpose_joints'].astype(np.float32)
                inp_bbox = self.skeleton_to_bbox(data_sample['inp_joints'])
                data_sample['inp_bbox_min_xyz'] = inp_bbox['min_xyz'].astype(np.float32)
                data_sample['inp_bbox_max_xyz'] = inp_bbox['max_xyz'].astype(np.float32)
                data_sample['inp_bbox_scale_xyz'] = (2 / (inp_bbox['max_xyz'] - inp_bbox['min_xyz'])).astype(np.float32)
                data_sample['obs_visible_vertices_mask']= smpl_vis_mask[inp_basename].astype(np.float32)
                data_sample['inp_camera'] = cameras[inp_basename]
                
                back_basename = int(inp_basename) + 180 if int(inp_basename) < 180 else int(inp_basename) - 180
                back_basename = '{:06d}'.format(back_basename)

                data_sample['back_framename'] = back_basename
                data_sample['back_maskname'] = back_basename
                data_sample['back_view_index'] = int(back_basename)
                data_sample['back_pose_index'] = 0
                data_sample['back_Rh'] = mesh_infos[back_basename]['Rh'].astype(np.float32)
                data_sample['back_Th'] = mesh_infos[back_basename]['Th'].astype(np.float32)
                data_sample['back_betas'] = mesh_infos[inp_basename]['betas'].astype(np.float32)
                data_sample['back_poses'] = mesh_infos[back_basename]['poses'].astype(np.float32)
                data_sample['back_joints'] = mesh_infos[back_basename]['joints'].astype(np.float32)
                data_sample['back_tpose_joints'] = mesh_infos[back_basename]['tpose_joints'].astype(np.float32)
                back_bbox = self.skeleton_to_bbox(data_sample['back_joints'])
                data_sample['back_bbox_min_xyz'] = back_bbox['min_xyz'].astype(np.float32)
                data_sample['back_bbox_max_xyz'] = back_bbox['max_xyz'].astype(np.float32)
                data_sample['back_bbox_scale_xyz'] = (2 / (back_bbox['max_xyz'] - back_bbox['min_xyz'])).astype(np.float32)
                data_sample['back_visible_vertices_mask']= smpl_vis_mask[back_basename].astype(np.float32)
                data_sample['back_camera'] = cameras[back_basename]
                data_sample['uvmap_gt_path'] = os.path.join(subj_dir, 'uvmap_gt.png')
                data_samples.append(data_sample)

        return data_samples

    def load_test_dataset(self, novel_pose_test=False, fast_validation=True, diff_angle_test=False):
        dataset_dir = self.dataset_path
        subjects = self.subjects
        
        data_samples = []

        if diff_angle_test:
            intv = 30
            obs_view_list = np.arange(0, 360, intv)
            fast_validation = False
        else:
            obs_view_list = self.test_obs_views 
        
        for ovidx, obs_view_index in enumerate(obs_view_list):

            for idx, subj_name in enumerate(subjects):
                # load frame name, camera data, mesh data, canonical joint data
                subj_dir = os.path.join(dataset_dir, f'{subj_name:04d}')
                
                with open(os.path.join(subj_dir, 'mesh_infos.pkl'), 'rb') as f:
                    mesh_infos = pickle.load(f)
                with open(os.path.join(subj_dir, 'cameras.pkl'), 'rb') as f:
                    cameras = pickle.load(f)
                with open(os.path.join(subj_dir, 'canonical_joints.pkl'), 'rb') as f:
                    canonical_joints = pickle.load(f)
                with open(os.path.join(subj_dir, 'smpl_vertex_visibility_mask.pkl'), 'rb') as f:
                    smpl_vis_mask = pickle.load(f)

                if idx == 0:
                    # This data will be used as common data for all subjects
                    self.canonical_joints = canonical_joints['joints']
                    self.canonical_bbox = self.skeleton_to_bbox(canonical_joints['joints'])
                
                framenames = np.array(sorted(list(mesh_infos.keys())))
                
                if diff_angle_test:
                    framenames_test = framenames[::intv]
                else:
                    framenames_test = framenames[::20] # 0, 20 ... 340, num of views : 18
                
                for jdx, frame_name in enumerate(framenames_test):
                    target_view_angle = int(frame_name)
                    view_index = target_view_angle
                    if view_index == obs_view_index:
                        continue
                    data_sample = {}
                    data_sample['db'] = cfg.db
                    data_sample['subject'] = f'{subj_name:04d}'
                    data_sample['view_index'] = view_index
                    data_sample['frame_name'] = frame_name # saved without extension
                    data_sample['pose_index'] = 0
                    
                    mesh_info = mesh_infos[frame_name]
                    bbox = self.skeleton_to_bbox(mesh_info['joints'])
                    
                    dst_skel_info = {}
                    dst_skel_info['betas'] = mesh_info['betas'].astype('float32')
                    dst_skel_info['poses'] = mesh_info['poses'].astype('float32')
                    dst_skel_info['dst_tpose_joints'] = mesh_info['tpose_joints'].astype('float32')
                    dst_skel_info['Rh'] = mesh_info['Rh'].astype('float32')
                    dst_skel_info['Th'] = mesh_info['Th'].astype('float32')
                    dst_skel_info['bbox'] = bbox # (min_xyz, max_xyz)

                    data_sample['dst_skel_info'] = dst_skel_info
                    data_sample['camera'] = cameras[frame_name]

                    # observation setting
                    inp_basename = framenames[obs_view_index]

                    data_sample['inp_framename'] = inp_basename
                    data_sample['inp_maskname'] = inp_basename
                    data_sample['obs_view_index'] = int(inp_basename)
                    data_sample['obs_pose_index'] = 0
                    data_sample['inp_Rh'] = mesh_infos[inp_basename]['Rh'].astype(np.float32)
                    data_sample['inp_Th'] = mesh_infos[inp_basename]['Th'].astype(np.float32)
                    data_sample['inp_betas'] = mesh_infos[inp_basename]['betas'].astype(np.float32)
                    data_sample['inp_poses'] = mesh_infos[inp_basename]['poses'].astype(np.float32)
                    data_sample['inp_joints'] = mesh_infos[inp_basename]['joints'].astype(np.float32)
                    data_sample['inp_tpose_joints'] = mesh_infos[inp_basename]['tpose_joints'].astype(np.float32)
                    inp_bbox = self.skeleton_to_bbox(data_sample['inp_joints'])
                    data_sample['inp_bbox_min_xyz'] = inp_bbox['min_xyz'].astype(np.float32)
                    data_sample['inp_bbox_max_xyz'] = inp_bbox['max_xyz'].astype(np.float32)
                    data_sample['inp_bbox_scale_xyz'] = (2 / (inp_bbox['max_xyz'] - inp_bbox['min_xyz'])).astype(np.float32)
                    data_sample['obs_visible_vertices_mask']= smpl_vis_mask[inp_basename].astype(np.float32)
                    data_sample['inp_camera'] = cameras[inp_basename]
                    back_basename = int(inp_basename) + 180 if int(inp_basename) < 180 else int(inp_basename) - 180
                    back_basename = '{:06d}'.format(back_basename)

                    data_sample['back_framename'] = back_basename
                    data_sample['back_maskname'] = back_basename
                    data_sample['back_view_index'] = int(back_basename)
                    data_sample['back_pose_index'] = 0
                    data_sample['back_Rh'] = mesh_infos[back_basename]['Rh'].astype(np.float32)
                    data_sample['back_Th'] = mesh_infos[back_basename]['Th'].astype(np.float32)
                    data_sample['back_betas'] = mesh_infos[inp_basename]['betas'].astype(np.float32)
                    data_sample['back_poses'] = mesh_infos[back_basename]['poses'].astype(np.float32)
                    data_sample['back_joints'] = mesh_infos[back_basename]['joints'].astype(np.float32)
                    data_sample['back_tpose_joints'] = mesh_infos[back_basename]['tpose_joints'].astype(np.float32)
                    back_bbox = self.skeleton_to_bbox(data_sample['back_joints'])
                    data_sample['back_bbox_min_xyz'] = back_bbox['min_xyz'].astype(np.float32)
                    data_sample['back_bbox_max_xyz'] = back_bbox['max_xyz'].astype(np.float32)
                    data_sample['back_bbox_scale_xyz'] = (2 / (back_bbox['max_xyz'] - back_bbox['min_xyz'])).astype(np.float32)
                    data_sample['back_visible_vertices_mask']= smpl_vis_mask[back_basename].astype(np.float32)
                    
                    data_sample['back_camera'] = cameras[back_basename]
                    data_sample['uvmap_gt_path'] = os.path.join(subj_dir, 'uvmap_gt.png')

                    data_samples.append(data_sample)
                
        if fast_validation:
            data_samples = data_samples[::5]
        
        return data_samples
    
    def get_total_frames(self):
        return len(self.datalist)


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx): # get single image's data component
        
        # Prepare data
        data = self.datalist[idx]
        results = { 'frame_name': data['frame_name'],
                    'subject': data['subject'],
                    'view_index': data['view_index'],
                    'pose_index': data['pose_index'],
                    'obs_view_index': data['obs_view_index'],
                    'obs_pose_index': data['obs_pose_index'],
                    'back_view_index': data['back_view_index'],
                    'back_pose_index': data['back_pose_index'],
                    }

        if self.bgcolor is None:
            # bgcolor = (np.random.rand(3) * 255.).astype('float32')
            bgcolor = np.array([0, 0, 0], dtype='float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')
        
        camera = data['camera']
        img_pil = self.load_image(self.dataset_path, data['subject'], 'images', data['frame_name'])
        alpha_pil = self.load_image(self.dataset_path, data['subject'], 'masks', data['frame_name'])
        img_black_pil, img_pil = self.mask_image(img_pil, alpha_pil, bgcolor)
        img = np.array(img_pil).astype(np.float32)/255.
        
        alpha = np.array(alpha_pil.convert('RGB')).astype('float32') / 255.
        
        if cfg.use_uv_inpainter:
            uv_map_gt = np.array(Image.open(data['uvmap_gt_path']).convert('RGB'))
            uv_map_gt = cv2.resize(uv_map_gt, (cfg.uv_map.uv_map_size, cfg.uv_map.uv_map_size), interpolation=cv2.INTER_LINEAR)
            uv_map_gt = uv_map_gt.astype(np.float32) / 255.
            results.update({'uv_map_gt': uv_map_gt})
        H, W = np.array(img_pil).shape[0:2] 
        
        dst_skel_info = data['dst_skel_info']

        info = {'img': np.array(img_pil).astype(np.float32)/255.,
                'alpha': alpha,
                'alpha_1ch': np.array(alpha_pil).astype(np.float32)/255.,
                'W': W,
                'H': H,
                'skel_info':dst_skel_info,
                'camera_dict':camera,
                'resize_img_scale':cfg.resize_img_scale,
                'ray_mode':self.ray_shoot_mode,
                'keyfilter':self.keyfilter}
        
        if self.ray_shoot_mode == 'patch':
            info.update({'N_patches':cfg.patch.N_patches,
                         'sample_subject_ratio':cfg.patch.sample_subject_ratio,
                         'patch_size':cfg.patch.size})
        target_ray_data = construct_ray_data(**info)
        
        # Observation motion base, canonical body data, motion weight priors, pose data
        dst_poses = dst_skel_info['poses']
        
        results.update({'img':img,
                        'alpha': alpha,
                        'img_width': W,
                        'img_height': H,
                        'bgcolor': bgcolor
                        })
        results.update({
                        'target_poses_69': dst_poses[3:],
                        'target_betas': dst_skel_info['betas']
                        })
        results.update(target_ray_data)
        
        
        # Preprocess of input image's data
        inp_frame_name = data['inp_framename']
        inp_camera = data['inp_camera']
        inp_img_pil = self.load_image(self.dataset_path, data['subject'], 'images', inp_frame_name)
        inp_alpha_pil = self.load_image(self.dataset_path, data['subject'], 'masks', inp_frame_name)
        inp_alpha = (np.array(inp_alpha_pil.convert('RGB')).astype('float32') / 255.)[...,[0]]
        inp_img_black_pil, inp_img_pil = self.mask_image(inp_img_pil, inp_alpha_pil, bgcolor)

        inp_E_data = inp_camera['extrinsics'].astype('float32')
        inp_K = inp_camera['intrinsics'].astype('float32')
        inp_K[:2] *= cfg.resize_img_scale

        inp_Rh = data['inp_Rh']
        inp_Th = data['inp_Th']

        inp_E = apply_global_tfm_to_camera(inp_E_data, inp_Rh, inp_Th).astype('float32')
        
        inp_img_normed = self.image_transform(inp_img_pil)[None]
        
        inp_betas = data['inp_betas']
        inp_poses = data['inp_poses']
        inp_joints = data['inp_joints']
        inp_bbox_min_xyz = data['inp_bbox_min_xyz']
        
        inp_bbox_scale_xyz = data['inp_bbox_scale_xyz']
        inp_H, inp_W = np.array(inp_img_pil).shape[0:2]

        # make 2d pose heatmap
        inp_joints_2d = self.load_2d_pose(inp_joints, inp_E, inp_K)
        inp_joints_2d_heatmap = self.generate_2d_heatmap(inp_joints_2d, (inp_W, inp_H)).astype(np.float32)
        heatmap_input = inp_joints_2d_heatmap

        obs_R = inp_E[:3, :3]
        obs_T = inp_E[:3, 3]
        inp_ray_o = -np.dot(obs_R.T, obs_T).ravel()

        # Observation data
        results.update({'inp_img':np.array(inp_img_pil).astype(np.float32)/255.,
                        'inp_alpha': inp_alpha,
                        'inp_img_width': W,
                        'inp_img_height': H,
                        'inp_extrinsics':inp_E,
                        'inp_intrinsics':inp_K,
                        'inp_img_normed':inp_img_normed,
                        'inp_heatmap':heatmap_input,
                        'inp_ray_o':inp_ray_o,
                        'inp_betas':inp_betas,
                        'inp_visible_vertices_mask':data['obs_visible_vertices_mask'],
                        'inp_bbox_min_xyz': inp_bbox_min_xyz,
                        'inp_bbox_scale_xyz': inp_bbox_scale_xyz,
                        'inp_poses_69': inp_poses[3:]
                        })
        
        # Preprocess of back image's data
        if cfg.back_net.back_net_on:
            back_frame_name = data['back_framename']
            back_img_pil = self.load_image(self.dataset_path, data['subject'], 'images', back_frame_name)
            back_alpha_pil = self.load_image(self.dataset_path, data['subject'], 'masks', back_frame_name)
            back_img_black_pil, back_img_pil = self.mask_image(back_img_pil, back_alpha_pil, bgcolor)

            back_img_normed = self.image_transform(back_img_pil)[None]
            
            back_camera = data['back_camera']
            back_E_data = back_camera['extrinsics'].astype('float32')
            back_K = back_camera['intrinsics'].astype('float32')
            back_K[:2] *= cfg.resize_img_scale

            back_Rh = data['back_Rh']
            back_Th = data['back_Th']

            back_E = apply_global_tfm_to_camera(back_E_data, back_Rh, back_Th).astype('float32')
            
        
            results.update({'back_extrinsics':back_E,
                            'back_intrinsics':back_K,
                            'back_img_gt':back_img_normed,
                            'back_visible_vertices_mask':data['back_visible_vertices_mask'],
                            })

        return results