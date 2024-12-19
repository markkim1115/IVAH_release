import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))
import numpy as np
import json
import pickle
import cv2
from third_parties.smpl.smpl import load_smpl_model, get_smpl_from_numpy_input
from tqdm import tqdm

root_dir = 'dataset/RenderPeople'
subject_dirs = [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
smpl_model = load_smpl_model(device='cpu')
with open(os.path.join(root_dir, 'human_list.txt'), 'r') as f:
    human_list = f.readlines()
    human_list = [x.strip() for x in human_list]

for subj_idx, subj in tqdm(enumerate(subject_dirs)):
    subj_root_dir = os.path.join(root_dir, subj)
    subj_img_root = os.path.join(subj_root_dir, 'img')
    subj_mask_root = os.path.join(subj_root_dir, 'mask')

    with open(os.path.join(subj_root_dir, 'cameras.json')) as jsonf:
        cameras = json.load(jsonf)
        views = np.arange(0,36)
        cam_keys = ['camera{:04d}'.format(idx) for idx in views]
    view_indexes_names = ['{:04d}'.format(vidx) for vidx in views]
    
    cam_dict = {}
    for vidx, view_index_name in enumerate(view_indexes_names):
        cam_dict[view_index_name] = {}
        
        R = np.array(cameras[cam_keys[vidx]]['R']).astype(np.float32)
        T = np.array(cameras[cam_keys[vidx]]['T']).astype(np.float32)
        E = np.eye(4)
        E[:3,:3] = R
        E[:3,3] = T

        K = np.array(cameras[cam_keys[vidx]]['K']).astype(np.float32)
        
        cam_dict[view_index_name]['extrinsics'] = E
        cam_dict[view_index_name]['intrinsics'] = K

    subj_smpl_root = os.path.join(subj_root_dir, 'outputs_re_fitting')
    smpl_fitting_data = dict(np.load(os.path.join(subj_smpl_root, 'refit_smpl_2nd.npz'), allow_pickle=True))
    smpl_data = smpl_fitting_data['smpl'].item()
    Rhs = smpl_data['global_orient'].astype(np.float32)
    body_poses = smpl_data['body_pose'].astype(np.float32)
    betas = smpl_data['betas'].astype(np.float32)
    Ths = smpl_data['transl'].astype(np.float32)
    # betas = np.zeros_like(betas).astype(np.float32)
    
    num_poses = len(Rhs)

    tposed_smpl_out = get_smpl_from_numpy_input(smpl_model, 
                                     global_orient=np.zeros((num_poses, 3)), 
                                     transl=np.zeros((num_poses, 3)), 
                                     body_pose=np.zeros_like(body_poses), 
                                     betas=betas, device='cpu')
    tpose_joints = tposed_smpl_out.joints.cpu().numpy().astype(np.float32)

    smpl_space_root = tpose_joints[0,0]
    
    betas = np.repeat(betas, num_poses, axis=0)


    smpl_out = get_smpl_from_numpy_input(smpl_model, 
                                     global_orient=np.zeros((num_poses, 3)), 
                                     transl=np.zeros((num_poses, 3)), 
                                     body_pose=body_poses, 
                                     betas=betas, device='cpu')
    joints = smpl_out.joints.cpu().numpy().astype(np.float32)
    
    mesh_infos = {}
    for pidx in range(num_poses):
        imgname = '{:04d}'.format(pidx)
        mesh_infos[imgname] = {}
        mesh_infos[imgname]['Rh'] = Rhs[pidx]
        Rh_mat_pidx = cv2.Rodrigues(Rhs[pidx])[0]
        
        # NOTE !!
        # Rh cannot be used directly to transform vertices from SMPL space to world space in HumanNeRF setting.
        # HumanNeRF rotates the camera pose instead of the human
        # So, we have to complement world translation to use global orientation vector to explicitly rotate SMPL-spaced body
        # New translation Th : root - root @ Rh_mat + Th

        mesh_infos[imgname]['Th'] = smpl_space_root - np.matmul(smpl_space_root, Rh_mat_pidx.T) + Ths[pidx]
        
        poses = np.zeros((72,), dtype=np.float32)
        poses[3:] = body_poses[pidx]
        mesh_infos[imgname]['poses'] = poses
        mesh_infos[imgname]['joints'] = joints[pidx]
        mesh_infos[imgname]['tpose_joints'] = tpose_joints[pidx]
        mesh_infos[imgname]['betas'] = betas[pidx]
    
    with open(os.path.join(subj_root_dir, 'mesh_infos.pkl'), 'wb') as f:
        pickle.dump(mesh_infos, f)
    
    with open(os.path.join(subj_root_dir, 'cameras.pkl'), 'wb') as f:
        pickle.dump(cam_dict, f)
    
    canonical_smpl = get_smpl_from_numpy_input(smpl_model, 
                                     global_orient=np.zeros((1, 3)), 
                                     transl=np.zeros((1, 3)), 
                                     body_pose=np.zeros((1, 69)), 
                                     betas=np.zeros((1,10)), device='cpu')
    
    template_joints = canonical_smpl.joints.cpu().numpy().astype(np.float32)[0]
    
    with open(os.path.join(subj_root_dir, 'canonical_joints.pkl'), 'wb') as f:   
        pickle.dump({'joints': template_joints}, f)