import os
import sys
import pickle
import numpy as np
import PIL.Image as Image
import time
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
from core.utils.camera_util import apply_global_tfm_to_camera
from core.utils.vis_util import draw_2D_joints
from third_parties.smpl.smpl import load_smpl_model, get_smpl_from_numpy_input
import cv2
from psbody.mesh.visibility import visibility_compute
import pdb
import shutil

dataset_root_zju = 'dataset/zju_mocap'
all_subject_list = [x for x in os.listdir(dataset_root_zju) if os.path.isdir(os.path.join(dataset_root_zju, x))]
all_subject_list.sort()

# Load smpl decoder
smpl_model = load_smpl_model('cuda')
faces = smpl_model.faces

for sidx, subject in enumerate(all_subject_list):
    data_dict = {}
    if not subject in ['CoreView_313', 'CoreView_315', 'CoreView_377']:
        continue
    
    subject_root = os.path.join(dataset_root_zju, subject)
    
    annots = np.load(os.path.join(subject_root, 'annots.npy'), allow_pickle=True).item()
    ims_annot = annots['ims']
    n_frames = len(ims_annot)
    cams = annots['cams']
    # camera infos
    Ks = np.array(cams['K']).astype(np.float32)
    Rs = np.array(cams['R']).astype(np.float32)
    Ts = np.array(cams['T']).astype(np.float32)
    Ds = np.array(cams['D']).astype(np.float32)
    n_views = len(Ks)
    
    for view_idx in range(n_views):
        data_dict[f'{view_idx:06d}'] = {}
        K = Ks[view_idx]
        R = Rs[view_idx]
        T = Ts[view_idx].flatten() / 1000.
        
        for frame_idx in range(n_frames):
            data_dict[f'{view_idx:06d}'][f'{frame_idx:06d}'] = np.zeros((6890,), dtype=bool)
            frame_number = frame_idx + 1 if subject in ['CoreView_313', 'CoreView_315'] else frame_idx
            filename = f'{frame_number}.npy'
            smpl_param = np.load(os.path.join(subject_root, 'new_params', filename), allow_pickle=True).item()
            Rh = smpl_param['Rh'][0]
            Th = smpl_param['Th'][0]
            betas = smpl_param['shapes'][0]
            poses = smpl_param['poses'][0]

            camera_position = (-R.T @ T)[None].astype(np.float64)
            smpl_out = get_smpl_from_numpy_input(smpl_model, global_orient=None, body_pose=poses[None,3:], betas=betas[None])
            vertices = smpl_out.vertices[0].cpu().numpy().astype(np.float64)
            Rh_mat = cv2.Rodrigues(Rh)[0]
            vertices = vertices@Rh_mat.T + Th

            visibility, _ = visibility_compute(v=vertices.astype(np.float64), f=faces.astype(np.uint32), cams=camera_position)
            visibility = visibility[0].astype(bool)
            data_dict[f'{view_idx:06d}'][f'{frame_idx:06d}'] = visibility
            print(f'{view_idx+1}/{n_views} | {frame_idx+1}/{n_frames} | {subject} visibility mask computing...')
    save_path = os.path.join(f'{subject_root}', 'smpl_vertex_visibility_mask.pkl')
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f'{sidx+1} / {len(all_subject_list)} | {subject} visibility mask dataset saved to {save_path}')