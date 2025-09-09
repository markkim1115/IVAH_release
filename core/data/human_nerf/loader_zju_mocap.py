import os
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data
from third_parties.smpl.smpl import load_smpl_model, get_smpl_from_numpy_input
from core.utils.camera_util import construct_ray_data, apply_global_tfm_to_camera
from configs import cfg
import torchvision.transforms as transforms
from PIL import Image
import pdb

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
            keyfilter=['rays', 'target_rgbs', 'dst_posevec_69'],
            maxframes=-1,
            bgcolor=None,
            ray_shoot_mode='image',
            target_view='0',
            fast_validation=True,
            novel_pose_test=False,
            **_):
        
        self.mode = mode
        train_subjects = [str(x) for x in [386, 387, 390, 392, 393, 394]]
        val_subjects = [str(x) for x in [313, 315, 377]]
        
        
        self.subjects = train_subjects if self.mode == 'train' else val_subjects
        self.dataset_path = 'dataset/zju_mocap'

        self.opt = Opt()
        self.image_transform = ImageTransform(self.opt)
        print('[Dataset Path]', self.dataset_path)
        self.novel_pose_test = novel_pose_test
        self.training_views = np.arange(0, 20)
        self.test_obs_views = [4,10,16]
        self.pose_interval = 5 if self.mode == 'train' else 20 # frame interval
        self.novel_view_pose_start = 0
        self.novel_pose_pose_start = 0
        self.num_pose = 100 if self.mode == 'train' else 25
        self.smpl_model = load_smpl_model('cpu')
        if self.mode == 'train':
            raise NotImplementedError
        
        else:
            (self.subject_names, 
                self.target_view_list, self.obs_view_list,
                self.target_pose_list, self.obs_pose_list,
                self.target_smpl_param_path_list, self.obs_smpl_param_path_list) = self.load_test_datalist(novel_pose_test, False)
            
        if self.mode == 'progress':
            indexes = np.array([14, 50, 78, 156, 157, 203, 310, 463, 
                                512, 553, 650,656,  697,  847, 1058, 
                                1141, 1153, 1248, 1294, 1309, 1486, 
                                1535, 1575, 1580, 1590, 1620, 1798, 
                                1819, 1939, 1979, 1992, 2007]) # 32 fixed sample
            if maxframes > 0:
                self.subject_names = self.subject_names[indexes]
                self.obs_view_list = self.obs_view_list[indexes]
                self.target_view_list = self.target_view_list[indexes]
                self.obs_pose_list = self.obs_pose_list[indexes]
                self.target_pose_list = self.target_pose_list[indexes]
                self.target_smpl_param_path_list = self.target_smpl_param_path_list[indexes]
                self.obs_smpl_param_path_list = self.obs_smpl_param_path_list[indexes]
                

            self.num_data_samples = len(indexes)
        
        elif self.mode == 'validation':
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

    def __len__(self):
        return self.num_data_samples
    
    def load_test_datalist(self, novel_pose_test=False, fast_validation=False):

        subject_names = []
        target_view_list = []
        obs_view_list = []
        target_pose_list = []
        obs_pose_list = []

        frame_infos_all = {}
        mask_infos_all = {}
        visibilities_all = {}
        cameras_all = {}

        for subject_name in self.subjects:
            subject_root = os.path.join(self.dataset_path, f'CoreView_{subject_name}')
            annots = np.load(os.path.join(subject_root, 'annots.npy'), allow_pickle=True).item()
            with open(os.path.join(subject_root, 'smpl_vertex_visibility_mask.pkl'), 'rb') as f:
                visibilities = pickle.load(f)
            visibilities_all[subject_name] = visibilities
            
            cam_annot = annots['cams']
            ims_annot = annots['ims']
            ims_annot = np.array([x['ims'] for x in ims_annot])[::self.pose_interval][:self.num_pose].transpose(1,0)
            n_frames = ims_annot.shape[-1]
            
            Ks = np.array(cam_annot['K']).astype(np.float32)
            Rs = np.array(cam_annot['R']).astype(np.float32)
            Ts = np.array(cam_annot['T']).astype(np.float32).reshape(-1, 3) / 1000.
            Ds = np.array(cam_annot['D']).astype(np.float32)
            
            Es = np.tile(np.eye(4, dtype=np.float32), (Rs.shape[0], 1, 1))
            Es[:, :3, :3] = Rs
            Es[:, :3, 3] = Ts
            cameras = {}
            for vidx in range(Rs.shape[0]):
                cameras[f'{vidx:06d}'] = {}
                cameras[f'{vidx:06d}']['extrinsics'] = Es[vidx]
                cameras[f'{vidx:06d}']['intrinsics'] = Ks[vidx]
            
            cameras_all[subject_name] = cameras
            frame_infos_all[subject_name] = ims_annot

        self.frame_infos_all = frame_infos_all
        self.mask_paths_all = mask_infos_all
        self.visibilities_all = visibilities_all
        self.cameras_all = cameras_all

        target_smpl_param_path_list = []
        obs_smpl_param_path_list = []

        test_view_indices = np.arange(0,20)[::2]
        for obs_view_idx in self.test_obs_views:
            for subject_name in self.subjects:
                subject_root = os.path.join(self.dataset_path, f'CoreView_{subject_name}')
                for target_view_idx in test_view_indices:
                    if not novel_pose_test and obs_view_idx == target_view_idx:
                        continue
                    frame_path_infos = frame_infos_all[subject_name][target_view_idx]
                    for frmidx, frm_path in enumerate(frame_path_infos):
                        target_frm_num = int(frm_path.split('/')[-1].split('_')[4]) if subject_name in ['313', '315'] else int(frm_path.split('/')[-1].split('.')[0])
                        target_pose_idx = frmidx
                        obs_pose_idx = target_pose_idx if not novel_pose_test else self.novel_pose_pose_start
                        obs_frm_path = frame_infos_all[subject_name][obs_view_idx][obs_pose_idx]
                        obs_frm_num = int(obs_frm_path.split('/')[-1].split('_')[4]) if subject_name in ['313', '315'] else int(obs_frm_path.split('/')[-1].split('.')[0])
                        if novel_pose_test and obs_pose_idx == target_pose_idx:
                            continue
                        target_smpl_param_path = os.path.join(subject_root, 'new_params', f'{target_frm_num}.npy')
                        obs_smpl_param_path = os.path.join(subject_root, 'new_params', f'{obs_frm_num}.npy')
                        # print(f'[{subject_name}] Obs View: {obs_view_idx}, Target View: {target_view_idx}, Obs Pose: {obs_pose_idx}, Target Pose: {target_pose_idx}')
                        
                        subject_names.append(subject_name)
                        target_view_list.append(target_view_idx)
                        obs_view_list.append(obs_view_idx)
                        target_pose_list.append(target_pose_idx)
                        obs_pose_list.append(obs_pose_idx)
                        target_smpl_param_path_list.append(target_smpl_param_path)
                        obs_smpl_param_path_list.append(obs_smpl_param_path)
        
        subject_names = np.array(subject_names)
        target_view_list = np.array(target_view_list)
        obs_view_list = np.array(obs_view_list)
        target_pose_list = np.array(target_pose_list)
        obs_pose_list = np.array(obs_pose_list)
        target_smpl_param_path_list = np.array(target_smpl_param_path_list)
        obs_smpl_param_path_list = np.array(obs_smpl_param_path_list)

        return subject_names, target_view_list, obs_view_list, target_pose_list, obs_pose_list, target_smpl_param_path_list, obs_smpl_param_path_list
    
    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

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
    
    def prepare_smpl_data(self, smpl_param_dict):
        Rh = smpl_param_dict['Rh'][0]
        Th = smpl_param_dict['Th'][0]
        poses = smpl_param_dict['poses'][0]
        betas = smpl_param_dict['shapes'][0]
        
        smpl_out = get_smpl_from_numpy_input(self.smpl_model, body_pose=poses[None,3:], betas=betas[None], device='cpu')
        joints = smpl_out.joints[0].cpu().numpy()
        verts = smpl_out.vertices[0].cpu().numpy()

        return Rh, Th, poses, betas, joints, verts
    
    def get_mask(self, mask_path):
        mask_cihp_path = mask_path

        mask_cihp = np.array(Image.open(mask_cihp_path))
        mask_cihp = (mask_cihp != 0).astype(np.uint8)
        
        mask_orig_path = mask_path.replace('mask_cihp', 'mask')
        mask_orig = np.array(Image.open(mask_orig_path))
        mask_orig = (mask_orig != 0).astype(np.uint8)

        mask = (mask_orig | mask_cihp).astype(np.uint8)
        mask[mask == 1] = 255

        return mask.astype(np.uint8)
    
    def __len__(self):
        return self.num_data_samples

    def __getitem__(self, idx): # get single image's data component
        
        # Currently, Only supports validation or progress mode
        if self.mode == 'train':
            raise NotImplementedError
        
        if self.mode != 'train':
            
            subject_name = self.subject_names[idx]
            view_idx = self.target_view_list[idx]
            obs_view_index = self.obs_view_list[idx]
            pose_idx = self.target_pose_list[idx]
            obs_pose_idx = self.obs_pose_list[idx]
            target_smpl_param_path = self.target_smpl_param_path_list[idx]
            obs_smpl_param_path = self.obs_smpl_param_path_list[idx]
            obs_frm_num = int(self.frame_infos_all[subject_name][obs_view_index][obs_pose_idx].split('/')[-1].split('_')[4]) if subject_name in ['313', '315'] else int(self.frame_infos_all[subject_name][obs_view_index][obs_pose_idx].split('/')[-1].split('.')[0])
            target_frm_num = int(self.frame_infos_all[subject_name][view_idx][pose_idx].split('/')[-1].split('_')[4]) if subject_name in ['313', '315'] else int(self.frame_infos_all[subject_name][view_idx][pose_idx].split('/')[-1].split('.')[0])    
            obs_frm_idx = obs_frm_num - 1 if subject_name in ['313', '315'] else obs_frm_num
            target_frm_idx = target_frm_num - 1 if subject_name in ['313', '315'] else target_frm_num
            frame_basename = os.path.basename(self.frame_infos_all[subject_name][view_idx][pose_idx])
            back_view_index = obs_view_index
            back_pose_idx = obs_pose_idx
        
        results = { 'frame_name': frame_basename,
                   'subject': subject_name,
                   'view_index': view_idx,
                   'pose_index': target_frm_idx,
                   'obs_view_index': obs_view_index,
                   'obs_pose_index': obs_frm_idx,
                   }
        bgcolor = np.array([0,0,0], dtype='float32')
        
        target_frame_path = self.frame_infos_all[subject_name][view_idx][pose_idx]
        image_path = os.path.join(self.dataset_path, f'CoreView_{subject_name}', target_frame_path)
        mask_path = os.path.join(self.dataset_path, f'CoreView_{subject_name}', 'mask_cihp', target_frame_path.replace('.jpg', '.png'))
        
        image_pil = Image.open(image_path)
        
        alpha = self.get_mask(mask_path)
        alpha_pil = Image.fromarray(alpha)
        _, masked_img_pil = self.mask_image(image_pil, alpha_pil, bgcolor)
        img = self.image_transform(masked_img_pil).permute(1,2,0).numpy()
        H, W = np.array(img).shape[0:2] 
        alpha = np.array(alpha_pil.convert('RGB')).astype(np.float32) / 255.
        alpha = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_NEAREST)
        
        if cfg.use_uv_inpainter:
            results.update({'uv_map_gt': np.zeros((H, W, 3), dtype=np.float32)})

        mesh_info = np.load(target_smpl_param_path, allow_pickle=True).item()
        Rh, Th, poses, betas, joints, verts = self.prepare_smpl_data(mesh_info)
        cameras = self.cameras_all[subject_name]
        bbox = self.skeleton_to_bbox(joints)

        dst_skel_info = {'betas': betas,
                         'poses': poses,
                         'Rh': Rh,
                         'Th': Th,
                         'bbox': bbox
                         }
        
        camera = cameras[f'{view_idx:06d}']
        
        info = {'img': img,
                'alpha': alpha,
                'alpha_1ch': alpha[..., [0]],
                'W': W,
                'H': H,
                'skel_info':dst_skel_info,
                'camera_dict':camera,
                'resize_img_scale':cfg.resize_img_scale,
                'ray_mode':self.ray_shoot_mode,
                'keyfilter':self.keyfilter}
        
        target_ray_data = construct_ray_data(**info)

        # Target pose data
        results.update({'img':img,
                        'alpha': alpha,
                        'img_width': W,
                        'img_height': H,
                        'bgcolor': bgcolor,
                        'target_poses_69': poses[3:],
                        'target_betas': betas,
                        
                        })
        
        results.update(target_ray_data)
        # target_E = apply_global_tfm_to_camera(camera['extrinsics'], Rh, Th).astype(np.float32)
        # target_K = camera['intrinsics'].copy()
        # target_K[:2] *= cfg.resize_img_scale
        # target_verts_2d = self.load_2d_pose(verts, target_E, target_K)
        # visibility_target = self.visibilities_all[subject_name][f'{view_idx:06d}'][f'{target_frm_idx:06d}'].astype(bool)
        # drawn_verts_target = draw_2D_joints(np.array(masked_img_pil.resize((512,512))), target_verts_2d[visibility_target])

        # Preprocess of input image's data
        obs_frame_path = self.frame_infos_all[subject_name][obs_view_index][obs_pose_idx]
        obs_image_path = os.path.join(self.dataset_path, f'CoreView_{subject_name}', obs_frame_path)
        obs_mask_path = os.path.join(self.dataset_path, f'CoreView_{subject_name}', 'mask_cihp', obs_frame_path.replace('.jpg', '.png'))
        
        obs_image_pil = Image.open(obs_image_path)
        obs_alpha = self.get_mask(obs_mask_path)
        obs_alpha_pil = Image.fromarray(obs_alpha.astype(np.uint8))
        _, obs_img_pil = self.mask_image(obs_image_pil, obs_alpha_pil, bgcolor)
        
        obs_alpha = (np.array(obs_alpha_pil.convert('RGB')).astype(np.float32)/255.)[...,[0]]
        obs_alpha = cv2.resize(obs_alpha, (W, H), interpolation=cv2.INTER_NEAREST)
        obs_img_normed = self.image_transform(obs_img_pil)[None]
        obs_img = obs_img_normed.permute(0,2,3,1).numpy()[0]
        obs_camera = cameras[f'{obs_view_index:06d}']
        obs_E_data = obs_camera['extrinsics']
        obs_K = obs_camera['intrinsics'].copy()
        obs_K[:2] *= cfg.resize_img_scale

        obs_smpl_param = np.load(obs_smpl_param_path, allow_pickle=True).item()
        obs_Rh, obs_Th, obs_poses, obs_betas, obs_joints, obs_verts = self.prepare_smpl_data(obs_smpl_param)

        obs_E = apply_global_tfm_to_camera(obs_E_data, obs_Rh, obs_Th).astype(np.float32)
        obs_visible_vertices_mask = self.visibilities_all[subject_name][f'{obs_view_index:06d}'][f'{obs_frm_idx:06d}']
        obs_visible_vertices_mask = obs_visible_vertices_mask.astype(np.float32)

        obs_bbox = self.skeleton_to_bbox(obs_joints)
        obs_bbox_min_xyz = obs_bbox['min_xyz'].astype(np.float32)
        obs_bbox_max_xyz = obs_bbox['max_xyz'].astype(np.float32)
        obs_bbox_scale_xyz = (2 / (obs_bbox_max_xyz - obs_bbox_min_xyz)).astype(np.float32)

        obs_H, obs_W = obs_img_normed.shape[2:]
        obs_R = obs_E[:3,:3]
        obs_T = obs_E[:3,3]
        obs_ray_o = -np.dot(obs_R.T, obs_T).ravel()

        # make 2d pose heatmap
        obs_joints_2d = self.load_2d_pose(obs_joints, obs_E, obs_K)
        # obs_verts_2d = self.load_2d_pose(obs_verts, obs_E, obs_K)
        # drawn_verts_obs = draw_2D_joints(np.array(obs_img_pil.resize((512,512))), obs_verts_2d[obs_visible_vertices_mask.astype(bool)])
        obs_joints_2d_heatmap = self.generate_2d_heatmap(obs_joints_2d, (W, H)).astype(np.float32)
        heatmap_input = obs_joints_2d_heatmap
        
        # Observation data
        results.update({'inp_img':obs_img,
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
            back_img_normed = torch.zeros_like(obs_img_normed).float()
            
            back_camera = cameras[f'{obs_view_index:06d}'].copy()
            back_E_data = back_camera['extrinsics'].astype('float32')
            back_K = back_camera['intrinsics'].astype('float32')
            back_K[:2] *= cfg.resize_img_scale

            back_Rh = obs_Rh
            back_Th = obs_Th
            back_E = apply_global_tfm_to_camera(back_E_data, back_Rh, back_Th).astype('float32')

            results.update({'back_extrinsics':back_E,
                            'back_intrinsics':back_K,
                            'back_img_gt':back_img_normed,
                            })
        # ray_mask = results['ray_mask'].reshape(H,W)
        # ray_mask = Image.fromarray((ray_mask*255).astype(np.uint8)).convert('RGB')
        # ray_mask = np.array(ray_mask)
        # from core.utils.vis_util import heatmap_to_jet
        # hmap = heatmap_to_jet(heatmap_input)
        # vis = np.concatenate([np.array(obs_img_pil.resize((512,512))), hmap, drawn_verts_obs, np.array(masked_img_pil.resize((512,512))), drawn_verts_target, ray_mask], axis=1)
        
        # save_dir = './zju_test_data/novel_pose' if self.novel_pose_test else './zju_test_data/novel_view'
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # Image.fromarray(vis).save(os.path.join(save_dir, f'{subject_name}_OBSVIEW_{obs_view_index}_TGTVIEW{view_idx}_OBSPOSE_{obs_pose_idx}_TGTPOSE_{pose_idx}.png'))
        # print(f'Save image: {subject_name}_OBSVIEW_{obs_view_index}_TGTVIEW{view_idx}_OBSPOSE_{obs_pose_idx}_TGTPOSE_{pose_idx}.png')
        return results
            
# dataset = Dataset(mode='validation', dataset_path='dataset/zju_mocap', novel_pose_test=False)
# for idx in range(len(dataset)):
#     res = dataset.__getitem__(idx)