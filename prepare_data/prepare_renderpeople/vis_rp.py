import pdb
import PIL.Image as Image
import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))
import numpy as np
import pickle
from core.utils.vis_util import draw_2D_joints, draw_3D_bbox_img, create_3d_figure, mesh_object
import cv2
from third_parties.smpl.smpl import load_smpl_model, get_smpl_from_numpy_input
from third_parties.smpl.smpl_numpy import SMPL as SMPL_NP
from tqdm import tqdm

def get_3D_bbox_points(joints):
    # joints: (24,3) in camera coordinate
    bbox_offset = 0.4
    min_xyz = np.min(joints, 0) - bbox_offset
    max_xyz = np.max(joints, 0) + bbox_offset
    
    bbox_dict = {'min_xyz':min_xyz, 'max_xyz':max_xyz}

    min_x, min_y, min_z = min_xyz[0], min_xyz[1], min_xyz[2]
    max_x, max_y, max_z = max_xyz[0], max_xyz[1], max_xyz[2]
    
    bbox_points = np.array([[min_x, min_y, min_z],
                            [min_x, min_y, max_z],
                            [min_x, max_y, min_z],
                            [min_x, max_y, max_z],
                            [max_x, min_y, min_z],
                            [max_x, min_y, max_z],
                            [max_x, max_y, min_z],
                            [max_x, max_y, max_z]])
    # bbox_points: (8,3)
    lines = np.array([[0,1],
                     [0,2],
                     [0,4],
                     [1,3],
                     [1,5],
                     [2,3],
                     [2,6],
                     [3,7],
                     [4,5],
                     [4,6],
                     [5,7],
                     [6,7]])

    
    return bbox_points, lines, bbox_dict

def apply_global_tfm_to_camera(E, Rh, Th):
    r""" Get camera extrinsics that considers global transformation.

    Args:
        - E: Array (3, 3)
        - Rh: Array (3, )
        - Th: Array (3, )
        
    Returns:
        - Array (3, 3)
    """
    global_tfms = np.eye(4)  #(4, 4)
    global_rot = cv2.Rodrigues(Rh)[0].T
    global_trans = Th
    global_tfms[:3, :3] = global_rot
    global_tfms[:3, 3] = -global_rot.dot(global_trans)
    return E.dot(np.linalg.inv(global_tfms))

def perspective_projection(R, T, K, pts):
    # pts [N,3]
    # R [3,3]
    # T [3,]
    # K [3,3]

    x_cam = np.matmul(pts, R.T) + T
    x = np.matmul(x_cam, K.T)
    x = x[:,:2] / x[:,2:]
    
    return x, x_cam

smpl_model = load_smpl_model(device='cpu')

data_root = '/media/cv1/data/RenderPeople_SHERF/20230228/'
subj_roots = [os.path.join(data_root,x) for x in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, x))]
# subj_roots = [subj_roots[400]]
for subj_root_idx, subj_root in tqdm(enumerate(subj_roots)):
    with open(os.path.join(subj_root, 'mesh_infos.pkl'), 'rb') as f:
        mesh_infos = pickle.load(f)
        f.close()

    with open(os.path.join(subj_root, 'cameras.pkl'), 'rb') as f:
        cameras = pickle.load(f)
        f.close()

    views = np.arange(0,36)
    view_index = 5
    pose_num = 10

    # load image
    img_dir = os.path.join(subj_root, 'img', 'camera{:04d}'.format(views[view_index]))
    img_path = os.path.join(img_dir, '{:04d}.jpg'.format(pose_num))
    img = Image.open(img_path)
    img = np.array(img)
    img = img[:,:,::-1]

    # get mesh info
    mesh_info = mesh_infos[f'{pose_num:04d}']
    Rh = mesh_info['Rh']
    Th = mesh_info['Th']
    body_pose = mesh_info['poses'][3:]
    betas = mesh_info['betas']
    Rh_mat = cv2.Rodrigues(Rh)[0]
    
    # SMPL in SMPL space
    smpl_space_smpl_out = get_smpl_from_numpy_input(smpl_model, 
                                        global_orient=None, 
                                        transl=None, 
                                        body_pose=body_pose[None], 
                                        betas=betas[None], device='cpu')
    
    smpl_space_joints = smpl_space_smpl_out.joints.cpu().numpy().astype(np.float32)[0]
    smpl_space_vertices = smpl_space_smpl_out.vertices.cpu().numpy().astype(np.float32)[0]

    joints = smpl_space_joints.copy()
    vertices = smpl_space_vertices.copy()

    # joints = np.matmul(smpl_space_joints, Rh_mat.T) + Th
    # vertices = np.matmul(smpl_space_vertices, Rh_mat.T) + Th

    # get camera info
    camera = cameras[f'{views[view_index]:04d}']
    R = camera['R']
    T = camera['T']
    K = camera['K']

    E = np.eye(4)
    E[:3,:3] = R
    E[:3,3] = T

    E = apply_global_tfm_to_camera(E, Rh, Th)
    R = E[:3,:3]
    T = E[:3,3]
    
    joints_2d, joints_cam = perspective_projection(R, T, K, joints)

    vertices_2d, vertices_cam = perspective_projection(R, T, K, vertices)
    vis = draw_2D_joints(img.copy(), vertices_2d[::20])[:,:,::-1]
    
    bbox_points, lines, bbox_dict = get_3D_bbox_points(joints_cam)
    vis = draw_3D_bbox_img(vis, bbox_points=bbox_points, bbox_lines=lines, intrinsic=K)
    Image.fromarray(vis).save('./{}.png'.format(os.path.basename(subj_root)))

    # vertex_colors = np.zeros_like(vertices)
    # # 3D visualization
    # objs = []
    # objs.append(mesh_object(vertices, smpl_model.faces, vertexcolor=vertex_colors, name='correct_world'))
    # print("\nWorld", vertices.mean())

    # vertex_colors = np.zeros_like(vertices) # Correct world vertices
    # vertex_colors += np.array([0, 0.2, 0.0])
    # objs.append(mesh_object(smpl_space_vertices, smpl_model.faces, vertexcolor=vertex_colors,name='smpl_space'))
    # print("smpl_space", smpl_space_vertices.mean())

    
    # vertex1 = np.matmul(smpl_space_vertices, Rh_mat.T)
    # vertex_colors = np.zeros_like(vertices)
    # vertex_colors += np.array([0, 0.4, 0.0])
    # print(vertex1.mean())
    # objs.append(mesh_object(vertex1, smpl_model.faces, vertexcolor=vertex_colors, name='mul_R'))

    # vertex2 = np.matmul(smpl_space_vertices, Rh_mat.T) + Th
    # vertex_colors = np.zeros_like(vertices)
    # vertex_colors += np.array([0.0, 0.6, 0.6])
    # print("vertex2", vertex2.mean())
    # objs.append(mesh_object(vertex2, smpl_model.faces, vertexcolor=vertex_colors, name='mul_R_add_T'))

    # vertex3 = np.matmul(smpl_space_vertices, Rh_mat.T) + np.matmul(Rh_mat, Th)
    # vertex_colors = np.zeros_like(vertices)
    # vertex_colors += np.array([0.6, 0.6, 0.6])
    # objs.append(mesh_object(vertex2, smpl_model.faces, vertexcolor=vertex_colors, name='mul_R_add_RhTh'))
    # print(vertex3.mean())

    # vertex4 = smpl_space_vertices - smpl_space_joints[0]
    # vertex4 = np.matmul(vertex4, Rh_mat.T) + smpl_space_joints[0] + Th
    # vertex_colors = np.zeros_like(vertices)
    # vertex_colors += np.array([0.6, 0.0, 0.0])
    # print('vertex4', vertex4.mean())
    # objs.append(mesh_object(vertex4, smpl_model.faces, vertexcolor=vertex_colors, name='remove_root_first'))

    # vertex5_Th = smpl_space_joints[0] - np.matmul(smpl_space_joints[0], Rh_mat.T) + Th
    # vertex5 = np.matmul(smpl_space_vertices, Rh_mat.T) + vertex5_Th
    # vertex_colors = np.zeros_like(vertices)
    # vertex_colors += np.array([0.3, 0.0, 0.9])
    # print('vertex5', vertex5.mean())
    # objs.append(mesh_object(vertex5, smpl_model.faces, vertexcolor=vertex_colors, name='calcualted'))

    # fig = create_3d_figure(objs)
    # fig.show()