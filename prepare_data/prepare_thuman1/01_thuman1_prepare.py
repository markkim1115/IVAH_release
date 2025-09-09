import numpy as np
import os, sys
import cv2
import pickle
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
from third_parties.smpl.smpl import load_smpl_model, get_smpl_from_numpy_input

# exclude_dirs = ['results_gyx_20181012_ym_2_F', 'results_gyx_20181013_lqj_1_F', 'results_gyx_20181013_yry_1_F', 'results_gyx_20181013_znn_1_F', 
#                         'results_gyx_20181013_zsh_1_M', 'results_gyx_20181013_zyj_1_F', 'results_gyx_20181014_sxy_2_F', 'results_gyx_20181015_dc_2_F', 
#                         'results_gyx_20181015_gh_2_F'] # pose number is less than 30

root_dir = '/media/cv1/T7/THuman_MPS_NeRF/nerf_data_'
subjects = [x for x in os.listdir(root_dir) if not 'txt' in x]
smpl_model = load_smpl_model()
tpose_smpl_ = get_smpl_from_numpy_input(smpl_model, np.zeros((1,3)).astype(np.float32), np.zeros((1,69)).astype(np.float32), np.zeros((1,10)).astype(np.float32), transl=np.zeros((1,3)).astype(np.float32))
tpose_joints = tpose_smpl_.joints.detach().numpy()[0]


for idx, subj in enumerate(subjects):
    files = os.listdir(os.path.join(root_dir, subj))
    
    annot = np.load(os.path.join(root_dir, subj, 'annots.npy'), allow_pickle=True).item()
    
    num_frames = len(annot['ims'])
    num_poses = num_frames
    mesh_infos = {}

    for poseidx in range(num_poses):
        fname = f'{poseidx}.npy'
        smpl_params = np.load(os.path.join(root_dir, subj, 'new_params', fname),allow_pickle=True).item()

        betas = smpl_params['shapes'] # (10,)
        Rhmat = smpl_params['R'] # (3,3)
        Rh = cv2.Rodrigues(Rhmat)[0].reshape(-1)
        body_pose = smpl_params['poses'].reshape(-1) # (72,1)
        Th = smpl_params['Th'].reshape(-1) # (1,3)

        smpl_ = get_smpl_from_numpy_input(smpl_model, np.zeros((1,3)).astype(np.float32), body_pose[3:].reshape(1,-1), betas[None], transl=np.zeros((1,3)).astype(np.float32))
        joints = smpl_.joints.detach().numpy()[0].reshape(-1,3)

        mesh_info_dict = {'Rh': Rh.copy(),
                          'Th': Th.copy(),
                          'betas': betas.copy(),
                          'poses': body_pose.copy(),
                          'joints': joints.copy(),
                          'tpose_joints': tpose_joints.reshape(-1,3).copy()}
        baseframename = f'{poseidx:06d}'
        mesh_infos[baseframename] = mesh_info_dict.copy()
    with open(os.path.join(root_dir, subj, 'mesh_infos.pkl'), 'wb') as f:
        pickle.dump(mesh_infos, f)
    
    ratio = int(idx/len(subjects) * 100)
    print(f'MESH_DATA : {subj} processed ({idx+1}/{len(subjects)}, {ratio}%)')

for idx, subj in enumerate(subjects):
    files = os.listdir(os.path.join(root_dir, subj))

    annot = np.load(os.path.join(root_dir, subj, 'annots.npy'), allow_pickle=True).item()

    cams = annot['cams']
    Ks = cams['K']
    Ds = cams['D']
    Rs = cams['R']
    Ts = cams['T']
    
    camera_dicts = {}
    for vidx in range(len(Ks)):
        K = Ks[vidx]
        D = Ds[vidx]
        R = Rs[vidx]
        T = Ts[vidx]
        E = np.eye(4)
        E[:3,:3] = R
        E[:3,3] = T.ravel()

        camera_dict = {'intrinsics': K.copy().astype(np.float32),
                       'distortion': D.copy().astype(np.float32),
                       'extrinsics': E.copy().astype(np.float32)}
        camera_dicts[f'{vidx:06d}'] = camera_dict.copy()
    with open(os.path.join(root_dir, subj, 'cameras.pkl'), 'wb') as f:
        pickle.dump(camera_dicts, f)
    ratio = int((idx+1)/len(subjects) * 100)
    print(f'CAMERA : {subj} processed ({idx+1}/{len(subjects)}, {ratio}%)')
print('done')