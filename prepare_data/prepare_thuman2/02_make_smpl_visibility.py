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
from psbody.mesh.visibility import visibility_compute
import pdb

dataset_root_thuman2 = 'dataset/thuman2'
all_subject_list = os.listdir(dataset_root_thuman2)
all_subject_list.sort()

## Load smpl mesh
smpl_model = load_smpl_model('cuda')
faces = smpl_model.faces

for subject in all_subject_list:
    subject_root = os.path.join(dataset_root_thuman2, subject)
    image_root = os.path.join(subject_root, 'images')
    angles = np.arange(0,360)
    mask_root = os.path.join(subject_root, 'masks')
    mesh_infos_path = os.path.join(subject_root, 'mesh_infos.pkl')
    cameras_root = os.path.join(subject_root, 'cameras.pkl')

    with open(mesh_infos_path, 'rb') as f:
        mesh_infos = pickle.load(f)
    with open(cameras_root, 'rb') as f:
        cameras = pickle.load(f)
    data_dict = {}
    for aidx in range(len(angles)):
        st = time.time()
        basename = '{:06d}'.format(angles[aidx])
        data_dict[basename] = np.zeros((6890,), dtype=bool)
        image_path = os.path.join(image_root, basename+'.png')
        mask_path = os.path.join(mask_root, basename+'.png')
        mask_pil = Image.open(mask_path)
        mask = np.array(mask_pil).astype(np.uint8)

        mesh_info = mesh_infos[basename]
        camera = cameras[basename]

        extrinsics = camera['extrinsics']
        K = camera['intrinsics']

        mesh_info = mesh_infos[basename]
        Rh = mesh_info['Rh']
        Th = mesh_info['Th']
        E = apply_global_tfm_to_camera(extrinsics, Rh, Th)
        R = E[:3, :3]
        T = E[:3, 3]
        camera_position = (-R.T @ T)[None].astype(np.float64)
        poses = mesh_info['poses']
        betas = mesh_info['betas']
        smpl_out = get_smpl_from_numpy_input(smpl_model, body_pose=poses[None, 3:], betas=betas[None])
        smpl_verts = smpl_out.vertices[0].cpu().numpy().astype(np.float64)
        smpl_faces = smpl_model.faces.astype(np.uint32)
        
        visibility, _ = visibility_compute(v=smpl_verts, f=smpl_faces, cams=camera_position)
        visibility = visibility.astype(bool)[0]
        
        debug = False
        if debug:
            from core.utils.vis_util import draw_3d_point_cloud, create_3d_figure
            vertices_np = smpl_verts[0].cpu().numpy()
            pc = draw_3d_point_cloud(vertices_np[visibility])
            create_3d_figure(pc).show()
            pdb.set_trace()
        
        data_dict[basename] = visibility
        et = time.time() - st
        print(f'Subject {subject} | view {aidx+1} / {len(angles)} | {et:.2f} sec/sample')
        
    save_path = os.path.join(f'{subject_root}', 'smpl_vertex_visibility_mask.pkl')
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f'{subject} visibility mask dataset saved to {save_path}')