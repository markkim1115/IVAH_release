import os
import sys
import pickle
import numpy as np
import PIL.Image as Image
import cv2
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
from core.utils.camera_util import apply_global_tfm_to_camera
from third_parties.smpl.smpl import load_smpl_model, get_smpl_from_numpy_input
from psbody.mesh.visibility import visibility_compute

dataset_root = 'dataset/itw_data'

## Load smpl mesh
smpl_model = load_smpl_model()
faces = smpl_model.faces

mesh_infos_path = os.path.join(dataset_root, 'mesh_infos.pkl')
cameras_root = os.path.join(dataset_root, 'cameras.pkl')

with open(mesh_infos_path, 'rb') as f:
    mesh_infos = pickle.load(f)
with open(cameras_root, 'rb') as f:
    cameras = pickle.load(f)
image_root = os.path.join(dataset_root, 'images')
mask_root = os.path.join(dataset_root, 'masks')

data_dict = {}

image_names = list(cameras.keys())
for i, image_name in enumerate(image_names):
    img_path = os.path.join(image_root, image_name+'.png')
    mask_path = os.path.join(mask_root, image_name+'_mask.png')
    
    camera = cameras[image_name]

    extrinsics = camera['extrinsics']
    K = camera['intrinsics']

    data_dict[image_name] = np.zeros((6890,), dtype=bool)

    image = Image.open(img_path)
    mask = Image.open(mask_path)
    image_size = image.size
    mask_size = mask.size
    assert image_size == mask_size
    
    image = np.array(image).astype(np.uint8)
    mask = np.array(mask).astype(np.uint8)

    mesh_info = mesh_infos[image_name]

    poses = mesh_info['poses']
    betas = mesh_info['betas']
    smpl_out = get_smpl_from_numpy_input(smpl_model, body_pose=poses[None, 3:], betas=betas[None])
    smpl_verts = smpl_out.vertices[0].cpu().numpy()
    smpl_joints = smpl_out.joints[0].cpu().numpy()
    smpl_faces = smpl_model.faces

    Rh = mesh_info['Rh']
    Rh_mat = cv2.Rodrigues(Rh)[0]
    cliff_trans = mesh_info['cliff_trans']
    tpose_smpl_out = get_smpl_from_numpy_input(smpl_model, body_pose=None, betas=mesh_info['betas'][None])
    tpose_joints = tpose_smpl_out.joints[0].cpu().numpy()
    tpose_root = tpose_joints[0]

    Th = -np.dot(tpose_root, Rh_mat.T) + tpose_root + cliff_trans # V_w = R(Vs-root) + root + trans => Rh = R, Th = -R(root) + root + trans
    
    mesh_info.update({'joints': smpl_joints, 'Th': Th})

    E = apply_global_tfm_to_camera(extrinsics, Rh, Th)
    R = E[:3, :3]
    T = E[:3, 3]
    camera_position = (-R.T @ T)[None].astype(np.float64)
    

    visibility, _ = visibility_compute(v=smpl_verts.astype(np.float64), f=smpl_faces.astype(np.uint32), cams=camera_position)
    visibility = visibility.astype(bool)[0]

    ## Mask out vertices that are not in human mask
    verts_cam = smpl_verts @ R.T + T
    verts_proj = verts_cam @ K.T
    verts_proj = verts_proj[..., :2] / verts_proj[..., 2:]

    image_plane_coord = verts_proj.astype(np.int32)

    if image_plane_coord[:, 0].max() >= image_size[0] or image_plane_coord[:, 1].max() >= image_size[1] or image_plane_coord[:, 0].min() < 0 or image_plane_coord[:, 1].min() < 0:
        point_mask = np.zeros((6890,), dtype=bool)
        for i in range(6890):
            if image_plane_coord[i, 0] < 0 or image_plane_coord[i, 0] >= image_size[0] or image_plane_coord[i, 1] < 0 or image_plane_coord[i, 1] >= image_size[1]:
                point_mask[i] = False
            else:
                point_mask[i] = mask[image_plane_coord[i, 1], image_plane_coord[i, 0]] > 0
        
    else:
        point_mask = mask[image_plane_coord[:, 1], image_plane_coord[:, 0]] > 0

    ## Final visibility mask
    # What i want to do is back-projecting image colors into vertices. We don't need vertices not in human mask.
    vertex_visibility_mask = np.logical_and(visibility, point_mask)

    debug = False
    if debug:
        from core.utils.vis_util import draw_3d_point_cloud, create_3d_figure
        vertices_np = smpl_verts
        pc = draw_3d_point_cloud(vertices_np[visibility])
        create_3d_figure(pc).show()
        breakpoint()

    data_dict[image_name] = visibility
    
    print(f'{image_name} | {i+1}/{len(image_names)}')

save_path = os.path.join(dataset_root, 'smpl_vertex_visibility_mask.pkl')
if os.path.exists(save_path):
    os.remove(save_path)
with open(save_path, 'wb') as f:
    pickle.dump(data_dict, f)
with open(os.path.join(dataset_root, 'mesh_infos.pkl'), 'wb') as f:
    pickle.dump(mesh_infos, f)
print(f'smpl vertex visibility mask dataset saved to {save_path}')