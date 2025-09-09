import pickle
import traceback
import pdb
import os
import sys
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
import gc
from prepare_data.prepare_thuman2.tools import *
from third_parties.smpl.smpl import load_smpl_model
from prepare_data.prepare_humman.render_ import render_images
from prepare_data.external.renderer.gl.init_gl import initialize_GL_context
from prepare_data.external.renderer.gl.prt_render import PRTRender

from prepare_data.external.prt_calc import PRT_calc

import trimesh
import PIL.Image as Image
import numpy as np
import cv2
import scipy.io as sio

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

    def search_mesh_obj(self, target_folder):
        obj_path = os.path.join(target_folder, 'mesh.obj')

        return obj_path

    def search_prt(self, target_folder):
        prt_path = [os.path.join(target_folder, x) for x in os.listdir(target_folder) if x.endswith('.mat')]
        if len(prt_path) > 0:
            return prt_path[0]
        else:
            return None

    def unit_process(self, target_folder, save_folder):
        """
        This function to search obj file, search UV texture file, make result folder and
        call functions for doing tasks indicated by --task option
        :param target_folder : ./THuman2.0/$file (eg. ./THuman2.0/0020)
        """
        egl = self.opt.egl
        ms_rate = self.opt.ms_rate
        width = 1920
        height = 1080
        
        initialize_GL_context(width=width, height=height, egl=egl)
        renderer = PRTRender(width=width, height=height, ms_rate=ms_rate, egl=self.opt.egl)
        
        framename_list = [x[:-4] for x in os.listdir(os.path.join(target_folder, 'smpl_params')) if x.endswith('.npz')]
        pose_list = framename_list
        view_names = [f'{i:06d}' for i in range(10)]
        # Load target Camera
        with open(os.path.join(target_folder, 'cameras.pkl'), 'rb') as f:
            cameras = pickle.load(f)
        for pidx, pose_name in enumerate(pose_list):
            if pidx > 18: continue
            # Load target Scan
            obj_path = os.path.join(target_folder, 'textured_meshes', f'{pose_name}.obj')
            tex_path = os.path.join(target_folder, 'textured_meshes', f'{pose_name}_0.png')
            mesh = None
            mesh = trimesh.load(obj_path)
            
            # Load target PRT
            pre_calculated_file = [f for f in os.listdir(os.path.join(target_folder, 'textured_meshes')) if f == f'{pose_name}_prt_data.mat']
            if len(pre_calculated_file) > 0 :
                print(f"pre-calculated PRT file detected: {pre_calculated_file}")
                try:
                    prt_mat = sio.loadmat(os.path.join(target_folder, 'textured_meshes', f'{pose_name}_prt_data.mat'))
                except:
                    if self.verbose: print(f"SUBJECT {subject} | POSE number {pose_name} PRT Calculation...")
                    if mesh is None: mesh = trimesh.load(obj_path, process=False, skip_materials=True, maintain_order=True)
                    prt_mat = PRT_calc(self.opt, mesh, obj_path)
                    sio.savemat(os.path.join(target_folder, 'textured_meshes',f'{pose_name}_prt_data.mat'), prt_mat, do_compression=True)
            elif len(pre_calculated_file) == 0:    
                if self.verbose: print(f"SUBJECT {subject} | POSE number {pose_name} PRT Calculation...")
                if mesh is None: mesh = trimesh.load(obj_path, process=False, skip_materials=True, maintain_order=True)
                prt_mat = PRT_calc(self.opt, mesh, obj_path)
                sio.savemat(os.path.join(target_folder, 'textured_meshes',f'{pose_name}_prt_data.mat'), prt_mat, do_compression=True)
            
            # Load target Scan
            mesh_obj = trimesh.load(obj_path)
            texture = cv2.imread(tex_path) if tex_path else None  # BGR
            render_images(self.opt, rndr=renderer, prt_mat=prt_mat, texture_image=texture, 
                            width=width, 
                            height=height, 
                            mesh=mesh_obj, 
                            obj_path=obj_path, 
                            cam_params=cameras, 
                            save_folder=save_folder, 
                            pose_name=pose_name)

            print(f"{subject} {pose_name+'obj'} | {pidx+1}/{len(pose_list)}")

            gc.collect()
        
if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser()

        parser.add_argument('--verbose', type=str2bool, default=True, help='print progress or not.')
        parser.add_argument('--subject', type=str, default='results_gyc_20181010_hsc_1_M')

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
        parser.add_argument('--task', type=list, default=['render'], nargs='+', help='preprocessing task selection')
        
        args = parser.parse_args()
        engine = TaskEngine(args)
        subject = str(args.subject)
        if 'p0004' in subject:
            exit(0)
        if 'p0005' in subject and int(subject.split('_')[0][1:]) < 595:
            exit(0)

        root_dir = '/media/cv1/T7/humman/recon'
        subject_dir = os.path.join(root_dir, f'{subject}')
        save_dir = f'{subject_dir}'+'/back_renderings'
        
        engine.unit_process(subject_dir, save_dir)
        