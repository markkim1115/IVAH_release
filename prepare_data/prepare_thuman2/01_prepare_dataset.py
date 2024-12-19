import pdb
import os
import sys
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
import shutil
import glob
import trimesh
import cv2
import pickle
import gc
import scipy.io as sio
from prepare_data.prepare_thuman2.tools import *
from third_parties.smpl.smpl import load_smpl_model, get_smpl_from_numpy_input
from prt import PRT_calc
from render_ import render_images
import numpy as np

class TaskEngine:
    """ This class is to initialize options. """
    def __init__(self, opt):
        """
        This function is to parse Engine Options
        :param opt: Engine Option
        """

        self.opt = opt
        self.tasks = opt.task
        self.verbose = opt.verbose
        self.smpl_model = load_smpl_model()
        
        print(f"Tasks: {opt.task}")

    @staticmethod
    def search_one_file(folder, ext):
        filepath = sorted(glob.glob(os.path.join(folder, f'*.{ext}')))[0]
        return filepath

    def search_obj(self, target_folder):
        """
        This function is to search obj files
        :param target_folder: ./THuman2.0/$file (eg. ./THuman2.0/0020)
        :return obj_path: eg. ./THuman2.0/0020/0020.obj
        """
        obj_path = self.search_one_file(target_folder, 'obj')
        return obj_path

    def search_prt(self, target_folder):
        prt_path = self.search_one_file(target_folder, 'mat')
        return prt_path

    def search_tex(self, target_folder, tex_ext='jpeg'):
        """
        This function is to search UV texture files
        :param target_folder : ./THuman2.0/$file (eg. ./THuman2.0/0020)
        :return tex_path: path of UV texture file - eg. ./THuman2.0/0020/material0.jpeg
        """
        try:
            tex_path = self.search_one_file(target_folder, tex_ext)
        except:
            tex_path = None
        return tex_path

    def load_smpl_data(self, subject, smpl_data_dir): # This may have problem.
        filename = os.path.join(smpl_data_dir, '{:04d}_smpl.pkl'.format(subject))
        smpl_ = pickle.load(open(filename, 'rb'), encoding='latin1')
        for k,v in smpl_.items():
            if v.ndim > 1:
                smpl_[k] = v[0]
                if k == 'body_pose':
                    smpl_[k] = smpl_[k].reshape(-1)
            else:
                smpl_[k] = v
        smpl_data = {}
        smpl_data['scale'] = smpl_['scale'].astype(np.float32)
        smpl_data['Rh'] = smpl_['global_orient'].astype(np.float32)
        global_rot_mat = cv2.Rodrigues(smpl_data['Rh'])[0]
        
        poses = np.zeros((72,)).astype('float32')
        poses[3:] = smpl_['body_pose'].reshape(-1)
        smpl_data['poses'] = poses
        smpl_data['betas'] = smpl_['betas'].reshape(-1)

        # get joints without global components
        smpl_out = get_smpl_from_numpy_input(self.smpl_model, np.zeros((1,3), dtype=np.float32), smpl_data['poses'][3:][None], smpl_data['betas'][None].astype(np.float32)) 
        joints = smpl_out.joints[0].detach().cpu().numpy()

        # tpose joints
        smpl_out = get_smpl_from_numpy_input(self.smpl_model, np.zeros((1,3), dtype=np.float32), np.zeros((1,69), dtype=np.float32), smpl_data['betas'][None].astype(np.float32)) 
        tpose_joints = smpl_out.joints[0].detach().cpu().numpy()
        cnl_joints = tpose_joints.copy()
        canonical_root_joint = cnl_joints[0]
        smpl_data['joints'] = joints
        smpl_data['tpose_joints'] = tpose_joints
        
        smpl_data['Th'] = canonical_root_joint - canonical_root_joint@global_rot_mat.T + np.array(smpl_['transl']).astype(np.float32) / smpl_['scale']
        return smpl_data, smpl_['scale'], cnl_joints, smpl_['transl'], canonical_root_joint
    
    def reorganize(self, smpl_data, smpl_cnl_joints, save_dir, handle_images=True, handle_mesh_data=True):
        if handle_images:
            
            source_img_path = os.path.join(save_dir, 'color_F' )
            target_img_path = os.path.join(save_dir, 'images')
            os.makedirs(target_img_path, exist_ok=True)
            shutil.copytree(source_img_path, target_img_path, dirs_exist_ok=True)
            
            source_mask_path = os.path.join(save_dir + '/mask_F', )
            target_mask_path = os.path.join(save_dir, 'masks')
            os.makedirs(target_mask_path, exist_ok=True)
            shutil.copytree(source_mask_path, target_mask_path, dirs_exist_ok=True)
            shutil.rmtree(source_img_path)
            shutil.rmtree(source_mask_path)
            
            source_img_path = os.path.join(save_dir, 'color_orth_B' )
            target_img_path = os.path.join(save_dir, 'images_orth_back')
            os.makedirs(target_img_path, exist_ok=True)
            shutil.copytree(source_img_path, target_img_path, dirs_exist_ok=True)
            
            source_mask_path = os.path.join(save_dir + '/mask_orth_B', )
            target_mask_path = os.path.join(save_dir, 'masks_orth_back')
            os.makedirs(target_mask_path, exist_ok=True)
            shutil.copytree(source_mask_path, target_mask_path, dirs_exist_ok=True)

            shutil.rmtree(source_img_path)
            shutil.rmtree(source_mask_path)
            os.remove(os.path.join(save_dir, 'material_0.jpeg'))
            os.remove(os.path.join(save_dir, 'material.mtl'))

        if handle_mesh_data:
            basenames = os.listdir(save_dir + '/images')
            self.generate_mesh_info_dict(smpl_data, basenames, save_dir)
            
            if os.path.exists(os.path.join(save_dir, 'canonical_joints.pkl')):
                os.remove(os.path.join(save_dir, 'canonical_joints.pkl'))
                p = os.path.join(save_dir, 'canonical_joints.pkl')
                print(f"Existing {p} removed.")

            with open(os.path.join(save_dir, 'canonical_joints.pkl'), 'wb') as f:
                pickle.dump({'joints' : smpl_cnl_joints}, f)
    
    def generate_mesh_info_dict(self, smpl_data, basenames, save_dir):
        mesh_infos = {}
        
        for idx in range(len(basenames)):
            mesh_infos[basenames[idx][:-4]] = smpl_data.copy()

        if os.path.exists(os.path.join(save_dir, 'mesh_infos.pkl')):
            os.remove(os.path.join(save_dir, 'mesh_infos.pkl'))
            p = os.path.join(save_dir, 'mesh_infos.pkl')
            print(f"Existing {p} removed.")

        with open(os.path.join(save_dir, 'mesh_infos.pkl'), 'wb') as f:
            pickle.dump(mesh_infos, f)
    
    def unit_process(self, target_folder, smpl_scale, smpl_transl, smpl_canonical_root_joint, save_folder, tex_ext='jpeg'):
        """
        This function to search obj file, search UV texture file, make result folder and
        call functions for doing tasks indicated by --task option
        :param target_folder : ./THuman2.0/$file (eg. ./THuman2.0/0020)
        """
        
        mesh = None
        obj_path = self.search_obj(target_folder)
        tex_path = self.search_tex(target_folder, tex_ext)
        result_folder = os.path.join(save_folder)
        os.makedirs(result_folder, exist_ok=True)
        
        if self.verbose:
            print(f"process: {target_folder}")
            print(f"object detect: {os.path.basename(obj_path)}")
            if tex_path:
                print(f"texture detect: {os.path.basename(tex_path)}")
            else:
                print(f"texture image undetected.")

        if 'normalize' in self.tasks:
            if self.verbose: print(f"normalize mesh...")
            mesh = trimesh.load(obj_path)
            
            mesh, obj_name, smpl_scale, rescale_param, norm_trans = mesh_normalize_by_smpl(self.opt, mesh, smpl_scale, smpl_transl, result_folder)
            obj_path = os.path.join(result_folder, obj_name)
        else:
            if False:
                obj_path = self.search_obj(result_folder)
                print(f"normalized object detect: {obj_path}")

        if 'prt' in self.tasks: 
            pre_calculated_file = [f for f in os.listdir(target_folder) if f == 'prt_data.mat']
            if len(pre_calculated_file) > 0 :
                print(f"pre-calculated PRT file detected: {pre_calculated_file}")
                prt_mat = sio.loadmat(self.search_prt(target_folder))
            else:    
                if self.verbose: print(f"PRT Calculation...")
                if mesh is None: mesh = trimesh.load(obj_path, process=False, skip_materials=True, maintain_order=True)
                prt_mat = PRT_calc(self.opt, mesh, obj_path)
                sio.savemat(target_folder+'/prt_data.mat',prt_mat,do_compression=True)

        if 'render' in self.tasks: 
            if self.verbose: print(f"Image Rendering...")
            texture = cv2.imread(tex_path) if tex_path else None  # BGR
            render_images(self.opt, obj_path, texture, prt_mat, result_folder)

        gc.collect()
        
        return rescale_param, norm_trans
    

if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser()

        parser.add_argument('--verbose', type=str2bool, default=True, help='print progress or not.')
        parser.add_argument('--subject', type=int, default=0)

        # PRT calculation options
        parser.add_argument('--folder', type=str, default=None, help='a target folder directory')
        
        parser.add_argument('--prt-n', dest='prt_n', type=int, default=40)
        parser.add_argument('--order', type=int, default=2)
        parser.add_argument('--num-processes', dest='num_processes', type=int, default=16,
                                 help='multi-process for PRT calc.')
        parser.add_argument('--image', type=str2bool, default=True)

        parser.add_argument('--egl', type=str2bool, default=False)
        parser.add_argument('--img-res', dest='img_res', type=int, default=512)
        parser.add_argument('--ms-rate', dest='ms_rate', type=float, default=1.0)
        parser.add_argument('--view-num', dest='view_num', type=int, default=360)
        parser.add_argument('--norm-overwrite', dest='norm_overwrite', type=str2bool, default=False, help='overwrite by normalized mesh(.obj) file or not.')
        # task selection
        parser.add_argument('--task', type=list, default=['normalize', 'prt', 'render'], nargs='+', help='preprocessing task selection')
        
        args = parser.parse_args()
        engine = TaskEngine(args)
        subject = args.subject
        
        thuman_dir = '/media/cv1/data/THuman2.0/meshes'
        root_dir = os.path.join(thuman_dir, '{:04d}'.format(subject))
        save_dir = 'dataset/thuman2/{:04d}'.format(subject)
        smpl_data, smpl_scale, cnl_joints, smpl_transl_raw, smpl_canonical_root_joint = engine.load_smpl_data(subject, os.path.join(os.path.dirname(thuman_dir), 'THuman2.0_smpl'))
        # rescale_param, norm_trans = engine.unit_process(root_dir, smpl_scale, smpl_transl_raw, smpl_canonical_root_joint, save_dir)
        engine.reorganize(smpl_data, cnl_joints, save_dir, handle_images=False, handle_mesh_data=True)