import traceback
import os
import sys
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
import gc
from prepare_data.prepare_thuman2.tools import *
from third_parties.smpl.smpl import load_smpl_model
from prepare_data.prepare_thuman1.render_ import render_images
from prepare_data.prepare_thuman2.external.renderer.gl.init_gl import initialize_GL_context
from prepare_data.prepare_thuman2.external.renderer.gl.color_render import ColorRender
import trimesh

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

    def fetch_mps_nerf_cam_params(self):
        # ========================== Cam Parameters ========================== #
        cam_param_path = '/media/cv1/T7/THuman_MPS_NeRF/nerf_data_/results_gyx_20181015_scw_2_M/annots.npy'
        annot = np.load(cam_param_path, allow_pickle=True).item()
        cams = annot['cams']
        Ks = cams['K']
        Rs = cams['R']
        Ts = cams['T']
        K = Ks[0]
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        params = {'Ks': Ks, 'Rs': Rs, 'Ts': Ts, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        return params
    
    def unit_process(self, target_folder, save_folder, pose_name_matching_dict):
        """
        This function to search obj file, search UV texture file, make result folder and
        call functions for doing tasks indicated by --task option
        :param target_folder : ./THuman2.0/$file (eg. ./THuman2.0/0020)
        """
        res = self.opt.img_res
        egl = self.opt.egl
        initialize_GL_context(width=res, height=res, egl=egl)
        renderer = ColorRender(width=self.opt.img_res, height=self.opt.img_res, egl=self.opt.egl)
        cam_params = self.fetch_mps_nerf_cam_params()
        pose_dirs = [x for x in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, x))]
        
        for pidx, pose_name in enumerate(pose_dirs):
            try:
                mps_pose_name = pose_name_matching_dict[pose_name]
                obj_path = self.search_mesh_obj(target_folder+'/{}'.format(pose_name))
                
                if self.verbose:
                    print(f"process: {pidx+1}/{len(pose_dirs)} | pose_name: {pose_name} -> mps_pose_name: {mps_pose_name}")
                    
                if 'render' in self.tasks: 
                    if self.verbose: print(f"Image Rendering...")
                    # Load target Scan
                    mesh_obj = trimesh.load(obj_path)
                    render_images(self.opt, renderer, mesh_obj, cam_params, save_folder, mps_pose_name, len(pose_dirs))
                
            except Exception as e:
                with open('prepare_data/prepare_thuman/error.txt', 'a') as f:
                    f.write(f"{subject} {pose_name} / {str(e)} : {traceback.format_exc()} \n")
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
        subject = args.subject
        
        thuman_dir = '/media/cv1/T7/THuman1_raw/dataset'
        root_dir = os.path.join(thuman_dir, '{}'.format(subject))
        mps_nerf_thuman_data_root = '/media/cv1/T7/THuman_MPS_NeRF/nerf_data_'
        save_dir = mps_nerf_thuman_data_root+'/{}'.format(subject)+'/back_renderings'
        pose_name_matching_file = f'/media/cv1/T7/THuman1_raw/matching_with_mps_nerf/{subject}.txt'
        if not os.path.exists(pose_name_matching_file):
            print(f"pose_name_matching_file not exists: {pose_name_matching_file}")
            sys.exit(1)
        pose_name_matching_dict = {}
        with open(pose_name_matching_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            mps_nerf_pose_name, pose_name = line.strip().split('/')
            pose_name_matching_dict[pose_name] = mps_nerf_pose_name
        
        engine.unit_process(root_dir, save_dir, pose_name_matching_dict)
        