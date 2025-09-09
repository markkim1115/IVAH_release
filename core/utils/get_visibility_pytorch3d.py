# This code generates visibility mask indices for SMPL vertices from given camera parameters and SMPL parameters using PyTorch3D.

import numpy as np
import PIL.Image as Image
import torch
import pickle
from core.utils.camera_util import apply_global_tfm_to_camera
from third_parties.smpl.smpl import get_smpl_from_numpy_input, load_smpl_model
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
from core.utils.vis_util import create_3d_figure, mesh_object, draw_3d_point_cloud, draw_3d_point, draw_3d_line

def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]

def get_camera_extrinsic(camera:dict, Rh, Th, device='cpu'):
    E = camera['extrinsics']
    # Rcam = E[:3,:3]
    # Tcam = E[:3,3]
    K = camera['intrinsics']

    E = apply_global_tfm_to_camera(E, Rh, Th)
    Rcam = E[:3,:3]
    Tcam = E[:3,3]

    Rcam_torch = torch.from_numpy(Rcam).float()[None]
    Tcam_torch = torch.from_numpy(Tcam).float()[None]
    K_torch = torch.from_numpy(K).float()[None]

    if device != 'cpu':
        Rcam_torch = Rcam_torch.cuda()
        Tcam_torch = Tcam_torch.cuda()
        K_torch = K_torch.cuda()
    return Rcam_torch, Tcam_torch, K_torch

def get_smpl_mesh(smpl_model, poses, betas, device='cpu'):
    body_pose = poses[3:]
    smpl_out = get_smpl_from_numpy_input(smpl_model, global_orient=None, body_pose=body_pose[None], betas=betas[None])
    smpl_verts = smpl_out.vertices
    
    smpl_verts_world = smpl_verts

    if device != 'cpu':
        smpl_verts_world = smpl_verts_world.cuda()
    faces = torch.from_numpy(smpl_model.faces.astype(int)).long().to(device)[None]
    
    return smpl_verts_world, faces

def vertex_transform_to_ndc(smpl_verts_world, Rcam_torch, Tcam_torch, K_torch, resolution_scale):
    # Convert world space vertices to camera space
    smpl_verts_cam = torch.matmul(Rcam_torch, smpl_verts_world.permute(0,2,1)).permute(0,2,1) + Tcam_torch
    
    # Project camera coordinate vertices to pixel space (For debugging)
    # smpl_verts_xyz = torch.matmul(K_torch, smpl_verts_world.permute(0,2,1)).permute(0,2,1)
    # smpl_verts_z_pix = smpl_verts_xyz[..., [2]]
    # smpl_verts_pix = smpl_verts_xyz[..., :2] / smpl_verts_xyz[..., 2:]
    
    base_resolution = 512
    resolution_scale = 1
    resolution = base_resolution * resolution_scale

    K_torch[:2] = K_torch[:2] * resolution_scale

    # Convert pixel space projection components to NDC space projection components
    f_ndc_x = K_torch[:,0,0]/resolution * 2
    f_ndc_y = K_torch[:,1,1]/resolution * 2
    c_ndc_x = (resolution/2 - K_torch[:,0,2]) / resolution * 2
    c_ndc_y = (resolution/2 - K_torch[:,1,2]) / resolution * 2
    K_torch_ndc = torch.tensor([[f_ndc_x, 0, c_ndc_x],
                                [0, f_ndc_y, c_ndc_y],
                                [0, 0, 1]]).float().to(smpl_verts_cam.device)
    
    # Project camera coordinate vertices to NDC space
    smpl_verts_xyz = torch.matmul(K_torch_ndc, smpl_verts_cam.permute(0,2,1)).permute(0,2,1)
    smpl_verts_z = smpl_verts_xyz[..., [2]]
    smpl_verts_xy_ndc = smpl_verts_xyz[..., :2] / smpl_verts_xyz[..., 2:]
    
    smpl_verts_ndc = torch.cat([smpl_verts_xy_ndc, smpl_verts_z], dim=-1)

    (x,y,z)=smpl_verts_ndc.split([1,1,1],dim=-1)
    
    xy = torch.cat([-x,-y],dim=-1) # flip x and y to rasterize in Pytorch3D NDC coordinate system
    xyz = torch.cat([xy,z],dim=-1)

    return xyz

def pytorch3D_mesh_object(verts, faces):
    meshes = Meshes(verts=verts, faces=faces)
    return meshes

def rasterize_mesh_(meshes:Meshes, image_size):
    pix_to_face, *_ = rasterize_meshes(meshes, image_size=image_size,
                        blur_radius=0.0,
                        faces_per_pixel=1,
                        )
    return pix_to_face

def get_visibility_mask(pix_to_face, faces):
    pix_idx = torch.unique_consecutive(pix_to_face)
    vis_vertices_id = torch.unique_consecutive(faces[0][pix_idx])
    vis_mask = torch.zeros(size=(6890,), device=faces.device)
    vis_mask[vis_vertices_id] = 1.
    vis_mask_np = vis_mask.cpu().numpy()
    return vis_mask, vis_mask_np

def visualize_raster_fragments(pix_to_face, image_path=None):
    pix_to_face_map = pix_to_face[0,...,-1].cpu().numpy()>=0
    pix_to_face_map = (pix_to_face_map.astype(np.float32)*255.0).astype(np.uint8)
    
    if image_path is not None:
        img = Image.open(image_path)
        img = np.array(img)
        pix_to_face_map = np.stack([pix_to_face_map, pix_to_face_map, pix_to_face_map], axis=-1)
        vis = np.concatenate([img, pix_to_face_map], axis=1)
    else:
        vis = pix_to_face_map
    Image.fromarray(vis).show()

def example():
    smpl_model = load_smpl_model()
    faces = torch.from_numpy(smpl_model.faces.astype(int)).long()
    
    subject_name = 'seq_000114-rp_corey_rigged_005'
    view_angle = 0

    frame_number = 17

    image_path = f'dataset/RenderPeople/{subject_name}/img/camera{view_angle:04d}/{frame_number:04d}.jpg'
    with open(f'dataset/RenderPeople/{subject_name}/cameras.pkl', 'rb') as f:
        cameras = pickle.load(f)
    with open(f'dataset/RenderPeople/{subject_name}/mesh_infos.pkl', 'rb') as f:
        mesh_infos = pickle.load(f)
    with open(f'dataset/RenderPeople/{subject_name}/smpl_vertex_visibility_mask.pkl', 'rb') as f:
        smpl_vertex_visibility_mask = pickle.load(f)
    
    camera = cameras[f'{view_angle:04d}']
    
    mesh_info = mesh_infos[f'{frame_number:04d}']
    
    # Ray casting based visibility annotations
    visibility_front = smpl_vertex_visibility_mask[f'{view_angle:04d}'][f'{frame_number:04d}']
    invisibility_front = ~visibility_front

    visibility_back = smpl_vertex_visibility_mask[f'{view_angle+18:04d}'][f'{frame_number:04d}']
    invisibility_back = ~visibility_back

    Rh = mesh_info['Rh']
    Th = mesh_info['Th']
    poses = mesh_info['poses']
    betas = mesh_info['betas']

    Rcam_torch, Tcam_torch, K_torch = get_camera_extrinsic(camera, Rh, Th, device='cuda')
    verts_world, faces = get_smpl_mesh(smpl_model, poses, betas, device='cuda')
    verts_ndc = vertex_transform_to_ndc(verts_world, Rcam_torch, Tcam_torch, K_torch, resolution_scale=1)
    meshes = pytorch3D_mesh_object(verts_ndc, faces)
    pix_to_face = rasterize_mesh_(meshes, image_size=512)
    visualize_raster_fragments(pix_to_face, image_path=image_path)
    vis_mask_torch, vis_mask_np = get_visibility_mask(pix_to_face, faces)

def debug():
    smpl_model = load_smpl_model()
    faces = torch.from_numpy(smpl_model.faces.astype(int)).long()
    
    subject_name = 'seq_000114-rp_corey_rigged_005'
    view_angle = 0

    frame_number = 17

    image_path = f'dataset/RenderPeople/{subject_name}/img/camera{view_angle:04d}/{frame_number:04d}.jpg'
    with open(f'dataset/RenderPeople/{subject_name}/cameras.pkl', 'rb') as f:
        cameras = pickle.load(f)
    with open(f'dataset/RenderPeople/{subject_name}/mesh_infos.pkl', 'rb') as f:
        mesh_infos = pickle.load(f)
    with open(f'dataset/RenderPeople/{subject_name}/smpl_vertex_visibility_mask.pkl', 'rb') as f:
        smpl_vertex_visibility_mask = pickle.load(f)
    
    camera = cameras[f'{view_angle:04d}']
    
    mesh_info = mesh_infos[f'{frame_number:04d}']
    
    # Ray casting based visibility annotations
    visibility_front = smpl_vertex_visibility_mask[f'{view_angle:04d}'][f'{frame_number:04d}']
    invisibility_front = ~visibility_front

    visibility_back = smpl_vertex_visibility_mask[f'{view_angle+18:04d}'][f'{frame_number:04d}']
    invisibility_back = ~visibility_back

    Rh = mesh_info['Rh']
    Th = mesh_info['Th']
    poses = mesh_info['poses']
    betas = mesh_info['betas']

    Rcam_torch, Tcam_torch, K_torch = get_camera_extrinsic(camera, Rh, Th, device='cuda')
    verts_world, faces = get_smpl_mesh(smpl_model, poses, betas, device='cuda')
    verts_ndc = vertex_transform_to_ndc(verts_world, Rcam_torch, Tcam_torch, K_torch, resolution_scale=1)
    meshes = pytorch3D_mesh_object(verts_ndc, faces)
    pix_to_face = rasterize_mesh_(meshes, image_size=512)
    visualize_raster_fragments(pix_to_face, image_path=image_path)
    vis_mask_torch, vis_mask_np = get_visibility_mask(pix_to_face, faces)

    vertex_pc_3d = draw_3d_point_cloud(verts_ndc.cpu().numpy()[0][vis_mask_np.astype(bool)], color='red')
    objs = [mesh_object(verts_world.cpu().numpy()[0], faces[0].cpu().numpy()), vertex_pc_3d]
    camera_origin = Rcam_torch.permute(0,2,1).matmul(-Tcam_torch[:,None].permute(0,2,1)).squeeze().cpu().numpy()
    cam_ori_obj = draw_3d_point(position=camera_origin)
    x_axis = draw_3d_line(start=camera_origin, end=camera_origin + Rcam_torch[0,0].cpu().numpy(), color='red')
    y_axis = draw_3d_line(start=camera_origin, end=camera_origin + Rcam_torch[0,1].cpu().numpy(), color='green')
    z_axis = draw_3d_line(start=camera_origin, end=camera_origin + Rcam_torch[0,2].cpu().numpy(), color='blue')
    objs.append(cam_ori_obj)
    objs.append(x_axis)
    objs.append(y_axis)
    objs.append(z_axis)

    fig = create_3d_figure(objs)
    fig.show()

if __name__ == '__main__':
    example()