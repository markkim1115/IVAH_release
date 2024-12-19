import os
import sys
import pickle
import numpy as np
import PIL.Image as Image
import glob
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
from core.utils.camera_util import apply_global_tfm_to_camera
from core.utils.vis_util import draw_2D_joints
from third_parties.smpl.smpl import load_smpl_model, get_smpl_from_numpy_input
from psbody.mesh.visibility import visibility_compute

dataset_root = 'dataset/thuman1_mpsnerf'
all_subject_list = [x for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))]
all_subject_list = sorted(all_subject_list)

## Load smpl mesh
smpl_model = load_smpl_model()
smpl_faces = smpl_model.faces

for sidx, subject in enumerate(all_subject_list):
    subject_dir = os.path.join(dataset_root, subject)
    mesh_infos_path = os.path.join(subject_dir, 'mesh_infos.pkl')
    with open(mesh_infos_path, 'rb') as f:
        mesh_infos = pickle.load(f)
    cam_param_path = os.path.join(subject_dir, 'annots.npy')
    annot = np.load(cam_param_path, allow_pickle=True).item()
    cams = annot['cams']
    Rs = cams['R']
    Ts = cams['T']
    Ks = cams['K']
    ims = annot['ims']
    num_poses = len(ims)

    data_dict = {}
    for vidx in range(24):
        data_dict[f'{vidx:06d}'] = {}

        image_dir = os.path.join(subject_dir, f'view_{vidx:03d}')
        mask_dir = os.path.join(subject_dir, 'mask_cihp', f'view_{vidx:03d}')
        num_poses = len(glob.glob(os.path.join(subject_dir,'new_params_neutral', '*.npy')))
        for pidx in range(num_poses):
            data_dict[f'{vidx:06d}'][f'{pidx:06d}'] = np.zeros((6890,), dtype=bool)
            image_path = os.path.join(image_dir, f'{pidx:06d}.png')
            mask_path = os.path.join(mask_dir, f'{pidx:06d}.png')
            image = Image.open(image_path)
            mask = Image.open(mask_path)
            image_size = image.size
            mask_size = mask.size
            assert image_size == mask_size

            image = np.array(image).astype(np.uint8)
            mask = np.array(mask).astype(np.uint8)

            E = np.eye(4)
            E[:3,:3] = Rs[vidx]
            E[:3,3] = Ts[vidx].ravel()
            K = Ks[vidx]
            mesh_info = mesh_infos[f'{pidx:06d}']
            Rh = mesh_info['Rh']
            Th = mesh_info['Th']

            E = apply_global_tfm_to_camera(E, Rh, Th.ravel())
            R = E[:3,:3]
            T = E[:3,3]
            camera_position = (-R.T @ T)[None].astype(np.float64)
            poses = mesh_info['poses']
            betas = mesh_info['betas']
            smpl_out = get_smpl_from_numpy_input(smpl_model, global_orient=None, body_pose=poses[None, 3:], betas=betas[None])
            vertices = smpl_out.vertices.detach().numpy()[0]

            visibility, _ = visibility_compute(v=vertices.astype(np.float64), f=smpl_faces.astype(np.uint32), cams=camera_position)
            visibility = visibility.astype(bool)[0]
            
            vertices_cam = vertices@R.T + T[None]
            vertices_2d = vertices_cam@K.T
            vertices_2d = vertices_2d[:,:2]/vertices_2d[:,2:]

            image_plane_coord = vertices_2d.astype(np.int32)
            
            if image_plane_coord[:, 0].max() >= image_size[0] or image_plane_coord[:, 1].max() >= image_size[1] or image_plane_coord[:, 0].min() < 0 or image_plane_coord[:, 1].min() < 0:
                point_mask = np.zeros((6890,), dtype=bool)
                for i in range(6890):
                    if image_plane_coord[i, 0] < 0 or image_plane_coord[i, 0] >= image_size[0] or image_plane_coord[i, 1] < 0 or image_plane_coord[i, 1] >= image_size[1]:
                        point_mask[i] = False
                    else:
                        point_mask[i] = mask[image_plane_coord[i, 1], image_plane_coord[i, 0]] > 0
                
            else:
                point_mask = mask[image_plane_coord[:, 1], image_plane_coord[:, 0]] > 0
            
            vertex_visibility_mask = np.logical_and(visibility, point_mask)
            
            debug = False
            if debug:
                from core.utils.vis_util import draw_3d_point_cloud, create_3d_figure, draw_2D_joints
                # vertices_np = vertices
                # pc = draw_3d_point_cloud(vertices_np[visibility])
                # create_3d_figure(pc).write_image(f'thuman1_{subject}_{vidx}_{pidx}_3d.png')
                vis = draw_2D_joints(image, vertices_2d[vertex_visibility_mask])
                Image.fromarray(vis).save(f'thuman1_{subject}_{vidx}_{pidx}_2d.png')
            
            data_dict[f'{vidx:06d}'][f'{pidx:06d}'] = vertex_visibility_mask
    
    save_path = os.path.join(f'{subject_dir}', 'smpl_vertex_visibility_mask.pkl')
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f'{sidx+1} / {len(all_subject_list)} | {subject} visibility mask dataset saved to {save_path}')
