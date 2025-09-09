import json
import os
import numpy as np
import cv2
import math
import random
import trimesh
import pickle
from prepare_data.external.renderer.gl.init_gl import initialize_GL_context
from prepare_data.external.renderer.gl.prt_render import PRTRender
from prepare_data.external.renderer.gl.color_render import ColorRender

from tqdm import tqdm
from prepare_data.external.renderer.mesh import load_obj_mesh, compute_tangent
from prepare_data.external.renderer.camera import Camera
from prepare_data.external.prt import sh_util

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


def remove_noise(mesh: trimesh.Trimesh, isolation_factor=0.2):
    verts = mesh.vertices
    faces = mesh.faces
    outlier_std = len(verts) * isolation_factor

    adj_faces = trimesh.graph.connected_components(mesh.face_adjacency, min_len=outlier_std)
    mask = np.zeros(len(faces), dtype=np.bool_)

    mask[np.concatenate(adj_faces)] = True

    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()



def load_render_param(param_dir=None):
    if param_dir is None:
        params = r'./params/render_param.json'
    else:
        params = param_dir

    with open(params) as f:
        param = json.load(f)

    return param


def cam_setting(param_dir=None):
    param = load_render_param(param_dir)
    img_res = param['img_res']
    focal_length = param['focal']
    near = param['near']
    far = param['far']
    cam_dist = param['cam_dist']  # 10
    view_num = param['view_num']

    # # ICON
    # focal_length = np.sqrt((img_res ** 2) * 2)

    # cam = Camera(width=img_res, height=img_res)
    cam = Camera(width=img_res, height=img_res, focal=focal_length, near=near, far=far)
    cam.sanity_check()
    cam_params = generate_cameras(cam_dist, view_num)

    return cam, cam_params


def render_images(opt, obj_path, texture_image, prt_mat, save_root):
    color = True if texture_image is not None else False
    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    egl = opt.egl  

    res = opt.img_res
    ms_rate = opt.ms_rate
    view_num = opt.view_num

    # Permissions
    image = opt.image
    
    # Light
    env_sh = np.load('prepare_data/prepare_thuman2/params/env_sh.npy')

    initialize_GL_context(width=res, height=res, egl=egl)

    # Target Save Folders
    makelist = ['color_F', 'mask_F', 'color_orth_B', 'mask_orth_B', 'normal_F', 'normal_orth_B', 'depth_F', 'depth_orth_B']
    for maketgt in makelist:
        os.makedirs(os.path.join(save_root, maketgt), exist_ok=True)
    
    # ========================= Calibration Parameters ========================= #
    # Get Camera List 0 ~ 359
    camera = Camera(width=res, height=res, focal=5000, near=0.1, far=40)
    camera.sanity_check()
    center = 0
    scale = 1
    # # ========================================================================== #

    # Load Pre-computed Radiance Transfer
    prt, face_prt = prt_mat['bounce0'], prt_mat['face']

    if color:
        # Load target Scan
        vertices, faces, normals, face_normals, textures, face_textures = load_obj_mesh(obj_path, True, True)
        # vertices = vertices * (1/smpl_scale) # Scale the dense 3D mesh to SMPL scale
        ydist = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        zdist = np.max(vertices[:, 2]) * 2
        zy_ratio = zdist / ydist
        print(f"ZY Ratio: {zy_ratio}")

        # Load Color Image
        texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

        # Renderer
        rndr = PRTRender(width=res, height=res, ms_rate=ms_rate, egl=egl)

        # Register
        rndr.set_norm_mat(scale, center)
        tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
        rndr.set_mesh(vertices, faces, normals, face_normals, textures, face_textures, prt, face_prt, tan, bitan)

        rndr.set_albedo(texture_image)
    else:
        # Load target Scan
        mesh = trimesh.load(obj_path)
        vertices = mesh.vertices
        # vertices = vertices * (1/smpl_scale)
        faces = mesh.faces
        textures = mesh.visual.vertex_colors[..., :3] / 255
        normals = mesh.vertex_normals

        ydist = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        zdist = np.max(vertices[:, 2]) * 2
        zy_ratio = zdist / ydist
        print(f"ZY Ratio: {zy_ratio}")

        rndr = ColorRender(width=res, height=res, egl=egl)
        rndr.set_mesh(vertices, faces, textures, normals)
        rndr.set_norm_mat(scale, center)

    sh_list = []
    cam_params = {}
    back_cam_params = {}
    for y in tqdm(range(0, 360, 360//view_num)):
        # ========================== Front side Phase ========================== #
        # ================= Calibration Matrix Export ================= #
        center, direction, right, up = generate_cameras(y, dist=22, smpl_canonical_root_joint=None)
        
        rot, trans = generate_cam_Rt(center, direction, right, up) # 여기가 잘못됬을 가능성도 있다.
        extrinsic = np.concatenate([rot, trans], axis=-1)  # [3x4]
        extrinsic = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], axis=0)
        
        K = np.eye(3)
        K[0,0], K[1,1] = 5000, 5000
        K[0,2], K[1,2] = res/2, res/2
        K = K.astype('float32') # 이건 틀릴 수가 없다. K는 고정.
        basename = '{:06d}'.format(int(y))
        cam_params[basename] = {}
        cam_params[basename]['extrinsics'] = extrinsic
        cam_params[basename]['intrinsics'] = K
        
        # ============================ 3 ============================== #

        # =========================== Location Set =========================== #
        camera.center = center
        camera.direction = direction
        camera.right = right
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

        rndr.set_sh(sh)
        rndr.analytic = False
        rndr.use_inverse_depth = False

        # ============================ 2 ============================ #

        # ========================== Render ========================== #
        rndr.display()

        object_name = f'{y:06d}.png'
        out_image = rndr.get_color(0) * 255
        out_mask = out_image[..., -1]

        if image:
            out_image = out_image[..., :3][..., ::-1]  # RGBA -> BGR
            cv2.imwrite(os.path.join(save_root, 'color_F', object_name), np.uint8(out_image))
            cv2.imwrite(os.path.join(save_root, 'mask_F', object_name), np.uint8(out_mask))
        if True:
            out_normal = rndr.get_color(1) * 255
            out_normal = out_normal[..., :3][..., ::-1]
            cv2.imwrite(os.path.join(save_root, 'normal_F', object_name), np.uint8(out_normal))
        if True:
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
        if image:
            out_image = rndr.get_color(0) * 255
            out_image = out_image[..., :3][..., ::-1]  # RGBA -> BGR
            cv2.imwrite(os.path.join(save_root, 'color_orth_B', object_name), np.uint8(out_image))
            cv2.imwrite(os.path.join(save_root, 'mask_orth_B', object_name), np.uint8(out_mask))
        if True:
            out_normal = rndr.get_color(1) * 255
            out_normal = out_normal[..., :3][..., ::-1]
            cv2.imwrite(os.path.join(save_root, 'normal_orth_B', object_name), np.uint8(out_normal))
        if True:
            back_depth = cv2.cvtColor(rndr.get_color(2), cv2.COLOR_RGBA2GRAY)
            back_valid = np.abs(back_depth[out_mask > 0])
            depth_norm = np.concatenate([front_valid, back_valid], axis=0)
            depth_norm = cv2.normalize(depth_norm, None, 0, 1, cv2.NORM_MINMAX)
            depth_norm[front_len:] = depth_norm[front_len:].max() - depth_norm[front_len:]  # back depth scale reverse
            depth_norm *= 255 * zy_ratio
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

    with open(os.path.join(save_root, 'cameras.pkl'), 'wb') as f:   
        pickle.dump(cam_params, f)