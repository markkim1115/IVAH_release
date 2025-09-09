import json
import os
import glob
import numpy as np
import cv2
import math
import trimesh
import pickle
import sys
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
from prepare_data.external.renderer.gl.init_gl import initialize_GL_context
from prepare_data.external.renderer.gl.color_render import ColorRender
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

def render_images(opt, rndr:ColorRender, mesh, cam_params:dict, save_folder:str, mps_pose_name:str, num_pose:int):
    mps_pose_name = int(mps_pose_name)
    save_pose_name = f'{mps_pose_name:06d}.png'
    # Light
    env_sh = np.load('prepare_data/prepare_thuman2/params/env_sh.npy')

    # ========================== Cam Parameters ========================== #
    Ks = cam_params['Ks']
    Rs = cam_params['Rs']
    Ts = cam_params['Ts']
    fx = cam_params['fx']
    fy = cam_params['fy']
    cx = cam_params['cx']
    cy = cam_params['cy']

    # ========================= Calibration Parameters ========================= #
    # Get Camera List 0 ~ 359
    camera = Camera(width=opt.img_res, height=opt.img_res, focal=fx, near=0.1, far=40)
    camera.sanity_check()
    center = 0
    scale = 1
    # # ========================================================================== #
    
    vertices = mesh.vertices
    faces = mesh.faces
    textures = mesh.visual.vertex_colors[..., :3] / 255
    normals = mesh.vertex_normals

    rndr.set_mesh(vertices, faces, textures, normals)
    rndr.set_norm_mat(scale, center)

    sh_list = []
    for vidx in tqdm(range(24)):
        # Target Save Folders
        back_color_save_dir = os.path.join(save_folder, 'images', f'view_{vidx:03d}')
        back_mask_save_dir = os.path.join(save_folder, 'masks', f'view_{vidx:03d}')

        # if len(glob.glob(os.path.join(back_color_save_dir, '*.png'))) == num_pose and len(glob.glob(os.path.join(back_mask_save_dir, '*.png'))) == num_pose:
        #     continue
        
        os.makedirs(back_color_save_dir, exist_ok=True)
        os.makedirs(back_mask_save_dir, exist_ok=True)

        # ========================== Front side Phase ========================== #
        # ================= Calibration Matrix Export ================= #
        E = np.eye(4)
        E[:3, :3] = Rs[vidx]
        E[:3, 3] = Ts[vidx].reshape(-1)
        center = -np.matmul(Rs[vidx].T, Ts[vidx])
        right = E[0,:3]
        up = -E[1,:3]
        direction = E[2,:3]
        rot, trans = generate_cam_Rt(center, direction, right, up)
        extrinsic = np.concatenate([rot, trans], axis=-1)  # [3x4]
        extrinsic = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], axis=0)
        
        # K = np.eye(3)
        # K[0,0], K[1,1] = 5000, 5000
        # K[0,2], K[1,2] = res/2, res/2
        # K = K.astype('float32') # 이건 틀릴 수가 없다. K는 고정.
        # basename = '{:06d}'.format(int(y))
        # cam_params[basename] = {}
        # cam_params[basename]['extrinsics'] = extrinsic
        # cam_params[basename]['intrinsics'] = K
        
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
        
        # rndr.display()
        object_name = f'{vidx:06d}.png'

        if False:
            out_image = rndr.get_color(0) * 255
            out_mask = out_image[..., -1]
            out_image = out_image[..., :3][..., ::-1]  # RGBA -> BGR
            cv2.imwrite(os.path.join(save_root, 'color_F', object_name), np.uint8(out_image))
            cv2.imwrite(os.path.join(save_root, 'mask_F', object_name), np.uint8(out_mask))
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
            out_image = out_image[..., :3][..., ::-1]  # RGBA -> BGR
            # flip image left, right
            out_image = cv2.flip(out_image, 1)
            cv2.imwrite(os.path.join(back_color_save_dir, save_pose_name), np.uint8(out_image))
            cv2.imwrite(os.path.join(back_mask_save_dir, save_pose_name), np.uint8(out_mask))
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