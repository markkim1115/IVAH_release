import json
import os
import glob
import numpy as np
import cv2
import PIL.Image as Image
import math
import trimesh
import pickle
import sys
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
from prepare_data.external.renderer.gl.prt_render import PRTRender
from prepare_data.external.renderer.mesh import load_obj_mesh, compute_tangent
import pdb
from tqdm import tqdm
from prepare_data.external.renderer.camera import Camera

def generate_cameras(angle, dist=10, view_num=360, smpl_canonical_root_joint=None):
    target = [0, 0, 0]
    up = [0, 1, 0]
    
    angle = (math.pi * 2 / view_num) * angle
    if smpl_canonical_root_joint is not None:
        smpl_cnl_root_y = smpl_canonical_root_joint[1]
        eye = np.asarray([dist * math.sin(angle), smpl_cnl_root_y, dist * math.cos(angle)])
    else:
        eye = np.asarray([dist * math.sin(angle), 0, dist * math.cos(angle)])

    fwd = np.asarray(target, np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    fwd[np.isnan(fwd)] = 0

    right = np.cross(fwd, up)  # 외적 함수
    right /= np.linalg.norm(right)
    right[np.isnan(right)] = 0

    down = np.cross(fwd, right)
    down[np.isnan(down)] = 0

    return eye, fwd, right, -down


def generate_cam_Rt(center, direction, right, up):
    def normalize_vector(v):
        v_norm = np.linalg.norm(v)
        return v if v_norm == 0 else v / v_norm

    center = center.reshape([-1])
    direction = direction.reshape([-1])
    right = right.reshape([-1])
    up = up.reshape([-1])

    rot_mat = np.eye(3)
    s = right
    s = normalize_vector(s)
    rot_mat[0, :] = s
    u = up
    u = normalize_vector(u)
    rot_mat[1, :] = -u
    rot_mat[2, :] = normalize_vector(direction)
    trans = np.expand_dims(-np.matmul(rot_mat, center), axis=-1)
    return rot_mat, trans

def render_images(opt, rndr:PRTRender, prt_mat, texture_image, width, height, mesh, obj_path, cam_params:dict, save_folder:str, pose_name):
    save_pose_name = f'{pose_name}.png'
    # Light
    env_sh = np.load('prepare_data/prepare_thuman2/params/env_sh.npy')
    # Texture image to RGB
    texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

    camera_objects = []
    extrinsic_list = []
    for vidx in range(10):
        
        # ========================== Cam Parameters ========================== #
        cam_param = cam_params[f'{vidx:06d}']
        K = cam_param['intrinsics']
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
    

        # ========================= Calibration Parameters ========================= #
        # Get Camera List 0 ~ 359
        camera = Camera(width=width, height=height, focal_x=fx, focal_y=fy, cx=cx, cy=cy, near=0.1, far=40)
        camera.sanity_check()
        center = 0
        scale = 1
        camera_objects.append(camera)
        extrinsic_list.append(cam_param['extrinsics'])
        # # ========================================================================== #
    for vidx in range(10):
        camera = camera_objects[vidx]
        center = 0
        scale = 1
        
        E = extrinsic_list[vidx]
        # # ========================== Load Pre-computed Radiance Transfer ========================== #
        # Load Pre-computed Radiance Transfer
        prt, face_prt = prt_mat['bounce0'], prt_mat['face']
        
        vertices, faces, normals, face_normals, textures, face_textures = load_obj_mesh(obj_path, True, True)
        vertices += np.zeros_like(vertices) + np.array([-0.01,0.02,0])[None]

        rndr.set_norm_mat(scale, center)
        tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
        rndr.set_mesh(vertices, faces, normals, face_normals, textures, face_textures, prt, face_prt, tan, bitan)
        rndr.set_albedo(texture_image)

        sh_list = []
        
        # Target Save Folders
        view_index_string = f'{vidx:03d}'
        front_color_save_dir = os.path.join(save_folder, '..', 'front_rendering', 'kinect_color', f'kinect_{view_index_string}')
        front_mask_save_dir = os.path.join(save_folder, '..', 'front_rendering', 'kinect_mask', f'kinect_{view_index_string}')
        back_color_save_dir = os.path.join(save_folder, 'kinect_color', f'kinect_{view_index_string}')
        back_mask_save_dir = os.path.join(save_folder, 'kinect_mask', f'kinect_{view_index_string}')

        # if len(glob.glob(os.path.join(back_color_save_dir, '*.png'))) == num_pose and len(glob.glob(os.path.join(back_mask_save_dir, '*.png'))) == num_pose:
        #     continue
        
        os.makedirs(front_color_save_dir, exist_ok=True)
        os.makedirs(front_mask_save_dir, exist_ok=True)
        os.makedirs(back_color_save_dir, exist_ok=True)
        os.makedirs(back_mask_save_dir, exist_ok=True)

        # ========================== Front side Phase ========================== #
        # ================= Calibration Matrix Export ================= #
        R = E[:3, :3]
        T = E[:3, 3]
        center = -np.matmul(R.T, T)
        right = E[0,:3]
        up = -E[1,:3]
        direction = E[2,:3]
        rot, trans = generate_cam_Rt(center, direction, right, up)
        extrinsic = np.concatenate([rot, trans], axis=-1)  # [3x4]
        extrinsic = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], axis=0)
        
        # ============================ 3 ============================== #

        # =========================== Location Set =========================== #
        camera.center = center
        camera.direction = direction
        camera.right = right
        camera.up = up
        camera.sanity_check()
        rndr.set_camera(camera)
        # ================================= 1 ================================ #

        # ====================== Random Light ====================== #
        # env_sh's shape = [240, 9, 3]
        # sh_id = random.randint(0, env_sh.shape[0] - 1)  # random choose [0 ~ 238]
        sh = env_sh[0]
        # sh_angle = 0.2 * np.pi * (random.random() - 0.5)
        sh_angle = 0.2 * np.pi
        # sh = sh_util.rotateSH(sh, sh_util.make_rotate(0, sh_angle, 0).T)
        sh_list.append(sh)

        rndr.analytic = False
        rndr.use_inverse_depth = False

        # rndr.set_sh(sh)
        rndr.analytic = False
        rndr.use_inverse_depth = False

        # ============================ 2 ============================ #

        # ========================== Render ========================== #
        rndr.display()

        object_name = f'{vidx:06d}.png'
        out_image = rndr.get_color(0) * 255
        out_mask = out_image[..., -1]

        if True:
            Image.fromarray(np.uint8(out_image[..., :3])).save(os.path.join(front_color_save_dir, object_name))
            Image.fromarray(np.uint8(out_mask)).save(os.path.join(front_mask_save_dir, object_name))
            
        if False:
            out_normal = rndr.get_color(1) * 255
            out_normal = out_normal[..., :3][..., ::-1]
            cv2.imwrite(os.path.join(save_root, 'normal_F', object_name), np.uint8(out_normal))
        if False:
            front_depth = cv2.cvtColor(rndr.get_color(2), cv2.COLOR_RGBA2GRAY)
            front_valid = front_depth[out_mask > 0]
            front_len = len(front_valid)
        # ========================== Back side Phase ========================== #
        # =========================== Location Set =========================== #
        camera.far, camera.near = camera.near, camera.far
        # camera.right = -camera.right  # Camera Space

        camera.sanity_check()
        rndr.set_camera(camera)
        # ================================= 5 ================================ #

        # ========================== Render ========================== #
        rndr.display()
        if True:
            out_image = rndr.get_color(0) * 255
            out_mask = out_image[..., -1]

            # flip image left, right to satisfy the perspective projection
            out_image = cv2.flip(out_image, 1)
            out_mask = cv2.flip(out_mask, 1)
            out_image = np.uint8(out_image)
            out_mask = np.uint8(out_mask)
            Image.fromarray(out_image[..., :3]).save(os.path.join(back_color_save_dir, save_pose_name))
            Image.fromarray(out_mask).save(os.path.join(back_mask_save_dir, save_pose_name))
            
        if False:
            out_normal = rndr.get_color(1) * 255
            out_normal = out_normal[..., :3][..., ::-1]
            cv2.imwrite(os.path.join(save_root, 'normal_orth_B', object_name), np.uint8(out_normal))
        if False:
            back_depth = cv2.cvtColor(rndr.get_color(2), cv2.COLOR_RGBA2GRAY)
            back_valid = np.abs(back_depth[out_mask > 0])
            depth_norm = np.concatenate([front_valid, back_valid], axis=0)
            depth_norm = cv2.normalize(depth_norm, None, 0, 1, cv2.NORM_MINMAX)
            depth_norm[front_len:] = depth_norm[front_len:].max() - depth_norm[front_len:]  # back depth scale reverse
            depth_norm *= 255
            front_valid = depth_norm[:front_len]
            back_valid = depth_norm[front_len:]

            front_depth[front_depth != 0] = front_valid.ravel()
            back_depth[back_depth != 0] = back_valid.ravel()

            cv2.imwrite(os.path.join(save_root, 'depth_F', object_name), np.uint8(front_depth))
            cv2.imwrite(os.path.join(save_root, 'depth_orth_B', object_name), np.uint8(back_depth))
        # ============================ 6 ============================= #
        camera.far, camera.near = camera.near, camera.far
        # camera.right = -camera.right  # Camera Space
        # ======================== Back side Phase End ======================== #
        print(f"Rendering views of a pose ... {vidx+1}/10")