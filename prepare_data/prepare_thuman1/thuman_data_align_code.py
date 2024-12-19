import os
import numpy as np
import glob
import pdb

def arrays_equal(a1, a2):
    return np.array_equal(a1, a2)

mps_root = '/media/cv1/T7/THuman_MPS_NeRF/nerf_data_'
raw_root = '/media/cv1/T7/THuman1_raw/dataset'
# From MPS NERF
all_subjects = [x for x in os.listdir(mps_root) if os.path.isdir(os.path.join(mps_root, x))]
save_dir = '/media/cv1/T7/THuman1_raw/matching_with_mps_nerf'
for sidx, subject in enumerate(all_subjects):
    mps_smpl_params_dir = os.path.join(mps_root, subject, 'new_params')
    mps_smpl_filenames = glob.glob(os.path.join(mps_smpl_params_dir, '*.npy'))
    mps_smpl_body_pose_dict = {}
    for pidx, smpl_param_file in enumerate(mps_smpl_filenames):
        mps_smpl_param = np.load(smpl_param_file, allow_pickle=True).item()
        mps_body_pose = mps_smpl_param['poses'].reshape(-1)[3:].astype(np.float64)
        mps_pose_name = os.path.basename(smpl_param_file).split('.')[0]
        mps_smpl_body_pose_dict[mps_pose_name] = [mps_body_pose]

    #From raw file
    subj_dir = os.path.join(raw_root, subject)
    raw_pose_names = [x for x in os.listdir(subj_dir) if os.path.isdir(os.path.join(subj_dir, x))]
    raw_pose_dict = {}
    for rawpidx, raw_pose_name in enumerate(raw_pose_names):
        with open(f'{raw_root}/{subject}/{raw_pose_name}/smpl_params.txt', 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        body_pose = np.array(lines[11:]).astype(np.float64)
        raw_pose_dict[raw_pose_name] = body_pose

    for k, v in mps_smpl_body_pose_dict.items():
        for k_raw, v_raw in raw_pose_dict.items():
            if arrays_equal(v[0], v_raw):
                print(f"Matched: {k} - {k_raw}")
                if not os.path.exists(f'{save_dir}/{subject}'):
                    os.makedirs(f'{save_dir}', exist_ok=True)
                with open(f'{save_dir}/{subject}.txt', 'a') as f:
                    f.write(f"{k}/{k_raw}\n")