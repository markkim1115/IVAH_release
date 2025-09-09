import numpy as np
import os, sys
import cv2
import pickle
import json
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
from third_parties.smpl.smpl import load_smpl_model, get_smpl_from_numpy_input

root_dir = '/media/cv1/T7/HuMMaN/recon'
subjects = [x for x in os.listdir(root_dir) if not 'txt' in x]
smpl_model = load_smpl_model()


n_views = 10
view_names = ['kinect_color_{:03d}'.format(vidx) for vidx in range(n_views)]

for idx, subj in enumerate(subjects):
    files = os.listdir(os.path.join(root_dir, subj))
    image_root = os.path.join(root_dir, subj, 'kinect_color')
    mask_root = os.path.join(root_dir, subj, 'kinect_mask')
    smpl_params_root = os.path.join(root_dir, subj, 'smpl_params')
    n_poses = len(os.listdir(smpl_params_root))
    frame_names = [x[:-4] for x in os.listdir(smpl_params_root)]
    
    with open(os.path.join(root_dir, subj, 'cameras.json'), 'r') as f:
        cameras = json.load(f)
    mesh_infos = {}
    camera_dict = {}
    for vidx in range(n_views):
        camera_dict[f'{vidx:06d}'] = {}
        camera = cameras['kinect_color_{:03d}'.format(vidx)]
        K = camera['K']
        R = camera['R']
        T = camera['T']
        K = np.array(K).reshape(3,3)
        R = np.array(R).reshape(3,3)
        T = np.array(T).reshape(3,1)
        E = np.eye(4)
        E[:3,:3] = R
        E[:3,3:] = T
        camera_dict[f'{vidx:06d}']['intrinsics'] = K.copy()
        camera_dict[f'{vidx:06d}']['extrinsics'] = E.copy()

        for pidx in range(n_poses):
            frame_name = frame_names[pidx]
            params = np.load(os.path.join(smpl_params_root, frame_name+'.npz'))
            betas = params['betas']
            tpose_smpl_ = get_smpl_from_numpy_input(smpl_model, np.zeros((1,3)).astype(np.float32), np.zeros((1,69)).astype(np.float32), betas=betas[None].astype(np.float32), transl=np.zeros((1,3)).astype(np.float32))
            tpose_joints = tpose_smpl_.joints.detach().numpy()[0]
            tpose_joint_root = tpose_joints[0].copy()
            body_pose = params['body_pose']
            global_orient = params['global_orient']
            Rh = global_orient
            Rh_mat = cv2.Rodrigues(global_orient)[0].astype(np.float32)
            transl = params['transl']
            transl = -Rh_mat@tpose_joint_root + tpose_joint_root + transl
            transl = transl[None]
            Th = transl[0]
            poses = np.zeros(72).astype(np.float32)
            poses[3:] = body_pose

            smpl_out = get_smpl_from_numpy_input(smpl_model, np.zeros((1,3)).astype(np.float32), body_pose[None], betas[None])
            joints = smpl_out.joints.detach().numpy()[0].reshape(-1,3)

            mesh_info_dict = {'Rh': Rh.copy(),
                          'Th': Th.copy(),
                          'poses': poses.copy(),
                          'betas': betas.copy(),
                          'joints': joints.copy(),
                          'tpose_joints': tpose_joints.reshape(-1,3).copy()}
            mesh_infos[frame_name] = mesh_info_dict.copy()
    
    with open(os.path.join(root_dir, subj, 'mesh_infos.pkl'), 'wb') as f:
        pickle.dump(mesh_infos, f)
    with open(os.path.join(root_dir, subj, 'cameras.pkl'), 'wb') as f:
        pickle.dump(camera_dict, f)
    ratio = int((idx+1)/len(subjects) * 100)
    print(f'{subj} processed ({idx+1}/{len(subjects)}, {ratio}%)')
