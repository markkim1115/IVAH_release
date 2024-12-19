import pdb

import os
import pickle
import glob
if __name__ == '__main__':
    import sys
    appendpath = os.path.abspath('.')
    sys.path.append(appendpath)
import numpy as np
import joblib
import torch
import torch.utils.data

from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes, query_dst_skeleton
from core.utils.file_util import list_files, split_path


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            mode,
            dataset_path,
            keyfilter=None,
            bgcolor=None,
            ray_shoot_mode='image',
            skip=1,
            maxframes=-1,
            **_):

        self.dataset_path = '/home/cv1/works/SymHumanNeRF/dataset/amass'
        
        test_annotfile_names = ['SSM_synced_db.pkl', 'Transitions_mocap_db.pkl']
        train_annotfile_names = [k for k in os.listdir(self.dataset_path) if not k in test_annotfile_names]
        self.annot_file_names = train_annotfile_names if mode == 'train' else test_annotfile_names
        print('[Dataset Path]', self.dataset_path)
        
        self.load_amass_motion_db()
        self.image_dir = os.path.join(dataset_path, 'images')
        self.mode = mode
        
        self.canonical_joints = self.db['template_joints'][0]
        self.canonical_bbox = self.skeleton_to_bbox(self.canonical_joints)

        self.keyfilter = keyfilter
        self.bgcolor = bgcolor

        self.ray_shoot_mode = ray_shoot_mode

    def load_amass_motion_db(self):
        print('Fetching AMASS motion database...')
        keys = ['Rh', 'poses', 'betas', 'Th', 'joints', 'tpose_joints', 'template_joints', 'motion_db', 'db_subject', 'vidname']
        db = dict.fromkeys(keys)
        for k in db.keys():
            db[k] = []

        for idx in range(len(self.annot_file_names)):
            annot = joblib.load(os.path.join(self.dataset_path, f'{self.annot_file_names[idx]}'))
            for k in keys:
                db[k].append(annot[k])
        for k in keys:
            if k not in ['motion_db', 'db_subject', 'vidname']:
                db[k] = np.concatenate(db[k], axis=0)
            else:
                db[k] = np.array(db[k])
        
        self.db = db.copy

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - 0.3
        max_xyz = np.max(skeleton, axis=0) + 0.3

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def __len__(self):
        return len(self.framelist)

    def __getitem__(self, idx): # get single image's data component
        # keys = ['Rh', 'poses', 'betas', 'Th', 'joints', 'tpose_joints']

        # Rh = self.db['Rh'][idx]
        poses = self.db['poses'][idx]
        # betas = self.db['betas'][idx]
        # Th = self.db['Th'][idx]
        joints = self.db['joints'][idx]
        tpose_joints = self.db['tpose_joints'][idx]
        # dst_bbox = self.skeleton_to_bbox(joints)
        dst_poses = poses
        dst_tpose_joints = tpose_joints
        results = {}

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(dst_poses, dst_tpose_joints)
            cnl_gtfms = get_canonical_global_tfms(self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms
            })

        if 'cnl_bbox' in self.keyfilter:
            canonical_bbox = self.canonical_bbox
            min_xyz = canonical_bbox['min_xyz'].astype('float32')
            max_xyz = canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
            })
        return results

if __name__ == '__main__':
    ds = Dataset('train', '/media/cv1/data/AMASS')
    pdb.set_trace()