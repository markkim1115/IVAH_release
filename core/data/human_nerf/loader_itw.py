import os
# import sys
# sys.path.append("/home/oem/members/dyub/SymHumanNeRF")
import pickle
import numpy as np
import torch
import torch.utils.data
from core.utils.camera_util import apply_global_tfm_to_camera, rotate_camera_by_frame_idx, get_rays_from_KRT, rays_intersect_3d_bbox
from configs import cfg
import torchvision.transforms as transforms
from PIL import Image
# from core.utils.vis_util import create_3d_figure, mesh_object, draw_camera_coordinate, draw_2D_joints, heatmap_to_jet, opencv_image_blending
# from third_parties.smpl.smpl import load_smpl_model, get_smpl_from_numpy_input
import cv2

def rotate_camera_around_y_axis(base_extrinsic, angle_degrees):
    """
    Rotates the camera around the Y-axis using the project's internal logic.
    This is a faithful implementation of `_update_extrinsics`.
    Args:
        base_extrinsic (np.ndarray): The base camera extrinsic (E2). (4x4)
        angle_degrees (float): The angle to rotate in degrees.
    Returns:
        np.ndarray: The new, rotated camera extrinsic matrix. (4x4)
    """
    angle_rad = np.deg2rad(angle_degrees)

    # Invert the view matrix to get the camera's world matrix
    inv_E = np.linalg.inv(base_extrinsic)

    # Extract camera's world rotation and position
    cam_rot_world = inv_E[:3, :3]
    cam_pos_world = inv_E[:3, 3]

    # Create the Y-axis rotation matrix
    grot_vec = np.array([0., angle_rad, 0.])
    grot_mtx = cv2.Rodrigues(grot_vec)[0].astype('float32')

    # Rotate the camera's position and orientation
    rotated_cam_pos = grot_mtx @ cam_pos_world
    rotated_cam_rot = grot_mtx @ cam_rot_world

    # Reassemble the new world matrix
    new_inv_E = np.eye(4)
    new_inv_E[:3, :3] = rotated_cam_rot
    new_inv_E[:3, 3] = rotated_cam_pos

    # Invert back to get the new view matrix
    new_E = np.linalg.inv(new_inv_E)

    return new_E.astype(np.float32)

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
    ROT_CAM_PARAMS = {
        'itw': {'rotate_axis': 'y', 'inv_angle': False}
    }
    
    def __init__(
            self, mode,
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            bgcolor=None,
            ray_shoot_mode='image',
            target_view='0',
            fast_validation=False,
            novel_pose_test=False,
            **_):
        
        self.mode = mode
        if self.mode == 'train':
            print("In the wild dataset is not available for training")
            exit()
        self.dataset_path = dataset_path
        self.image_transform = ImageTransform(Opt())
        print('[Dataset Path]', self.dataset_path)
        self.canonical_joints, self.canonical_bbox = None, None
        self.load_test_dataset(self.dataset_path)
        self.render_frames = 12
        self.keyfilter = keyfilter
        self.bgcolor = bgcolor
        self.novel_pose_test = novel_pose_test
        self.ray_shoot_mode = ray_shoot_mode
        if self.novel_pose_test:
            self.target_pose_basenames = np.random.choice(list(self.mesh_infos.keys()), size=len(self.mesh_infos.keys()))
        print(f' -- Total Validation Frames: {len(self.mesh_infos.keys())*self.render_frames}')

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - 0.4
        max_xyz = np.max(skeleton, axis=0) + 0.4

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

    def load_test_dataset(self, dataset_path):
        with open(os.path.join(dataset_path, 'mesh_infos.pkl'), 'rb') as f:
            self.mesh_infos = pickle.load(f)
        with open(os.path.join(dataset_path, 'cameras.pkl'), 'rb') as f:
            self.cameras = pickle.load(f)
        with open(os.path.join(dataset_path, 'smpl_vertex_visibility_mask.pkl'), 'rb') as f:
            self.smpl_vis_mask = pickle.load(f)
        
    def __len__(self):
        return len(self.mesh_infos.keys()) * self.render_frames

    def get_freeview_camera(self, input_extrinsics, frame_idx, total_frames, trans=None):
        E = rotate_camera_by_frame_idx(
                extrinsics=input_extrinsics, 
                frame_idx=frame_idx,
                period=total_frames,
                trans=trans,
                **self.ROT_CAM_PARAMS['itw'])
        return E

    def __getitem__(self, idx): # get single image's data component
        
        # Prepare data
        subject_idx = idx // self.render_frames
        view_idx = idx % self.render_frames
        framenames = np.array(sorted(list(self.mesh_infos.keys())))
        frame_name = framenames[subject_idx]
        
        bgcolor = np.array(self.bgcolor, dtype='float32')
        target_pose_basename = self.target_pose_basenames[subject_idx]
        results = { 'frame_name': frame_name,
                    'subject': frame_name,
                    'view_index': view_idx,
                    'pose_index': 0,
                    'obs_view_index': 0,
                    'obs_pose_index': 0,
                    'back_view_index': 0,
                    'back_pose_index': 0,
                    }
                    
        inp_img_pil = Image.open(os.path.join(self.dataset_path, 'images', frame_name+'.png'))
        inp_alpha_pil = Image.open(os.path.join(self.dataset_path, 'masks', frame_name+'_mask.png'))
        inp_alpha = (np.array(inp_alpha_pil.convert('RGB')).astype('float32') / 255.)[...,[0]]
        _, inp_img_pil = self.mask_image(inp_img_pil, inp_alpha_pil, np.array([0, 0, 0], dtype='float32'))

        inp_img_normed = self.image_transform(inp_img_pil)[None]
        
        inp_mesh_info = self.mesh_infos[frame_name]
        inp_betas = inp_mesh_info['betas'].astype('float32')
        inp_poses = inp_mesh_info['poses'].astype('float32')
        inp_joints = inp_mesh_info['joints'].astype('float32')
        inp_H, inp_W = np.array(inp_img_pil).shape[0:2]
        
        inp_camera = self.cameras[frame_name]

        inp_E_data = inp_camera['extrinsics'].astype('float32')
        inp_K = inp_camera['intrinsics'].astype('float32')
        inp_Rh = inp_mesh_info['Rh'].astype('float32')
        inp_Th = inp_mesh_info['Th'].astype('float32')

        inp_E = apply_global_tfm_to_camera(inp_E_data, inp_Rh, inp_Th).astype('float32')
        obs_R = inp_E[:3, :3]
        obs_T = inp_E[:3, 3]
        inp_ray_o = -np.dot(obs_R.T, obs_T).ravel()

        # make 2d pose heatmap
        inp_joints_2d = self.load_2d_pose(inp_joints, inp_E, inp_K)
        inp_joints_2d_heatmap = self.generate_2d_heatmap(inp_joints_2d, (inp_W, inp_H)).astype(np.float32)
        
        # Observation data
        results.update({'inp_img':np.array(inp_img_pil).astype(np.float32)/255.,
                        'inp_alpha': inp_alpha,
                        'inp_img_width': inp_W,
                        'inp_img_height': inp_H,
                        'inp_extrinsics':inp_E,
                        'inp_intrinsics':inp_K,
                        'inp_img_normed':inp_img_normed,
                        'inp_heatmap':inp_joints_2d_heatmap,
                        'inp_ray_o':inp_ray_o,
                        'inp_betas':inp_betas,
                        'inp_visible_vertices_mask':self.smpl_vis_mask[frame_name].astype(np.float32),
                        'inp_poses_69': inp_poses[3:]
                        })
        
        img_size = int(512 * cfg.resize_img_scale)
        img = np.zeros((img_size, img_size, 3)).astype(np.float32)
        alpha = np.zeros((img_size, img_size, 3)).astype(np.float32)
        alpha_1ch = np.zeros((img_size, img_size, 1)).astype(np.float32)
        
        if cfg.use_uv_inpainter:
            uv_map_gt = np.zeros((cfg.uv_map.uv_map_size, cfg.uv_map.uv_map_size, 3)).astype(np.float32)
            results.update({'uv_map_gt': uv_map_gt})
        
        Rh = inp_mesh_info['Rh'].astype('float32')
        Th = inp_mesh_info['Th'].astype('float32')
        joints = inp_mesh_info['joints'].astype('float32')
        if self.novel_pose_test:
            joints = self.mesh_infos[target_pose_basename]['joints'].astype('float32')
        bbox = self.skeleton_to_bbox(joints)
        
        target_camera = inp_camera.copy()
        
        if self.ray_shoot_mode == 'patch':
            assert False, "Patch mode(training) is not supported for ITW dataset"
        
        dst_bbox = bbox # dict, contains {minxyz, maxxyz} got from 3D skeleton
    
        K = target_camera['intrinsics'][:3, :3].copy().astype(np.float32)
        K[:2] *= cfg.resize_img_scale

        E_data = target_camera['extrinsics'].copy().astype(np.float32)
        E = apply_global_tfm_to_camera(
                E=E_data,
                Rh=Rh,
                Th=Th).astype(np.float32)
        
        angle = idx * (360.0 / self.render_frames)
        E = rotate_camera_around_y_axis(E, angle)
        R = E[:3, :3]
        T = E[:3, 3]
            
        rays_o, rays_d = get_rays_from_KRT(img_size, img_size, K, R, T)
        ray_img = img.reshape(-1, 3) 
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        x,y,w,h = cv2.boundingRect(ray_mask.reshape(img_size,img_size).astype(np.uint8))
        tight_ray_mask = [x,y,w,h]
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]
        ray_alpha = alpha_1ch.reshape(-1)[ray_mask]
        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')
        batch_rays = np.stack([rays_o, rays_d], axis=0)
        
        
        if 'rays' in self.keyfilter:
            target_ray_data = {}
            target_ray_data.update({
                'extrinsics':E,
                'intrinsics':K,
                'ray_mask': ray_mask,
                'tight_ray_mask': tight_ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'target_rgbs': ray_img,
                'target_alpha': ray_alpha
                })
            results.update(target_ray_data)
        
        
        dst_poses = inp_mesh_info['poses']
        dst_betas = inp_mesh_info['betas']
        
        if self.novel_pose_test:
            dst_poses = self.mesh_infos[target_pose_basename]['poses'].astype('float32')
        
        results.update({'img':img,
                        'alpha': alpha,
                        'img_width': img_size,
                        'img_height': img_size,
                        'bgcolor': bgcolor,
                        'target_poses_69': dst_poses[3:],
                        'target_betas': dst_betas
                        })
        
        # Preprocess of back image's data
        if cfg.back_net.back_net_on:
            back_img_normed = np.zeros((1, 3, img_size, img_size)).astype(np.float32)
            
            back_camera = inp_camera
            back_E_data = back_camera['extrinsics'].astype('float32')
            back_K = back_camera['intrinsics'].astype('float32')
            back_K[:2] *= cfg.resize_img_scale

            back_Rh = inp_Rh
            back_Th = inp_Th

            back_E = apply_global_tfm_to_camera(back_E_data, back_Rh, back_Th).astype('float32')
            
            results.update({'back_extrinsics':back_E,
                            'back_intrinsics':back_K,
                            'back_img_gt':back_img_normed,
                            'back_visible_vertices_mask':self.smpl_vis_mask[frame_name].astype(np.float32),
                            })

        return results

if __name__ == '__main__':
    dataset = Dataset(mode='test', dataset_path='dataset/itw_data', keyfilter=cfg.test_keyfilter)
    for i in range(len(dataset)):
        print(dataset[i])
        