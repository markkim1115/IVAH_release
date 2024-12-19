import os
import sys
import pdb
import traceback
import pickle
import numpy as np
import PIL.Image as Image
import glob
import time
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
from core.utils.camera_util import apply_global_tfm_to_camera
from core.utils.vis_util import draw_2D_joints
from third_parties.smpl.smpl import load_smpl_model, get_smpl_from_numpy_input
from psbody.mesh.visibility import visibility_compute

dataset_root_RP = 'dataset/humman'
all_subject_list = [x for x in os.listdir(dataset_root_RP) if os.path.isdir(os.path.join(dataset_root_RP, x))]
all_subject_list = sorted(all_subject_list)

## Load smpl mesh
smpl_model = load_smpl_model()
faces = smpl_model.faces

for sidx, subject in enumerate(all_subject_list):
    # if os.path.exists(os.path.join(f'{dataset_root_RP}/{subject}', 'smpl_vertex_visibility_mask.pkl')):
    #     print(f'{subject} already has visibility mask dataset')
    #     continue
    subject_root = os.path.join(dataset_root_RP, subject)
    mesh_infos_path = os.path.join(subject_root, 'mesh_infos.pkl')
    cameras_root = os.path.join(subject_root, 'cameras.pkl')
    with open(mesh_infos_path, 'rb') as f:
        mesh_infos = pickle.load(f)
    with open(cameras_root, 'rb') as f:
        cameras = pickle.load(f)

    
    image_root = os.path.join(subject_root, 'kinect_color')
    mask_root = os.path.join(subject_root, 'kinect_mask')
    data_dict = {}
    smpl_params_root = os.path.join(subject_root, 'smpl_params')
    frame_names = [x[:-4] for x in os.listdir(smpl_params_root) if x.endswith('.npz')]
    frame_names.sort()
    
    for viewidx in range(10):
        data_dict[f'{viewidx:06d}'] = {}

        view_wise_img_root = os.path.join(image_root, 'kinect_'+f'{viewidx:03d}')
        view_wise_mask_root = os.path.join(mask_root, 'kinect_'+f'{viewidx:03d}')
        image_paths = sorted(glob.glob(os.path.join(view_wise_img_root, '*.png')))
        mask_paths = sorted(glob.glob(os.path.join(view_wise_mask_root, '*.png')))

        num_poses = len(frame_names)
        assert num_poses == len(mask_paths)
        camera = cameras[f'{viewidx:06d}']
        
        extrinsics = camera['extrinsics']
        K = camera['intrinsics']
        
        for index in range(num_poses):
            st = time.time()
            frame_name = frame_names[index]
            
            data_dict[f'{viewidx:06d}'][frame_name] = np.zeros((6890,), dtype=bool)

            image_path = image_paths[index]
            mask_path = mask_paths[index]
            pose_idx = index
            try:
                image = Image.open(image_path)
                mask = Image.open(mask_path)
                image_size = image.size
                mask_size = mask.size
                assert image_size == mask_size
                
                image = np.array(image).astype(np.uint8)
                mask = np.array(mask).astype(np.uint8)
                mesh_info = mesh_infos[frame_name]
                Rh = mesh_info['Rh']
                Th = mesh_info['Th']
                E = apply_global_tfm_to_camera(extrinsics, Rh, Th)
                R = E[:3, :3]
                T = E[:3, 3]
                camera_position = (-R.T @ T)[None].astype(np.float64)
                poses = mesh_info['poses']
                betas = mesh_info['betas']
                
                smpl_out = get_smpl_from_numpy_input(smpl_model, body_pose=poses[None,3:], betas=betas[None])
                smpl_verts = smpl_out.vertices[0].cpu().numpy()
                smpl_faces = smpl_model.faces
                
                visibility, _ = visibility_compute(v=smpl_verts.astype(np.float64), f=smpl_faces.astype(np.uint32), cams=camera_position)
                visibility = visibility.astype(bool)[0]
            except Exception as e:
                with open("humman_visibility_error.txt", "a") as f:
                    f.write(f'{subject} {viewidx} {frame_name} {str(e)} {traceback.format_exc()}\n')
                continue
            
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
                from core.utils.vis_util import draw_3d_point_cloud, create_3d_figure, draw_2D_joints
                # vertices_np = vertices
                # pc = draw_3d_point_cloud(vertices_np[visibility])
                # create_3d_figure(pc).write_image(f'thuman1_{subject}_{vidx}_{pidx}_3d.png')
                vis = draw_2D_joints(image, verts_proj[vertex_visibility_mask])
                Image.fromarray(vis).save(f'thuman1_{subject}_{viewidx}_{frame_name}_2d.png')
                continue
            
            data_dict[f'{viewidx:06d}'][frame_name] = visibility.copy()
            et = time.time() - st

            print(f'Subject {subject} {sidx+1}/{len(all_subject_list)}| view {viewidx+1} / 10 | pose {index+1} / {num_poses} | {et:.2f} sec/sample')
            
        
    save_path = os.path.join(f'{subject_root}', 'smpl_vertex_visibility_mask.pkl')
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f'{subject} visibility mask dataset saved to {save_path}')