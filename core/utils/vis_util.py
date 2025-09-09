import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from typing import NamedTuple
import cv2
import PIL.Image as Image

def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))

def extract_rgb_values(color_list):
    color_list = [c[4:-1] for c in color_list]
    color_list = np.array([c.split(',') for c in color_list]).astype(int)

    return color_list.tolist()

def create_color_palette_Light24():
    color_palette_hex = plotly.colors.qualitative.Light24
    color_palette_rgb = np.array([hex_to_rgb(c) for c in color_palette_hex]) # (24,3)

    return color_palette_hex , color_palette_rgb

class Lighting(NamedTuple):  # pragma: no cover
    ambient: float = 0.5
    diffuse: float = 1.0
    fresnel: float = 0.0
    specular: float = 0.0
    roughness: float = 0.5
    facenormalsepsilon: float = 1e-6
    vertexnormalsepsilon: float = 1e-12

def mesh_object(verts, faces, vertexcolor=None, opacity=1.0, name=None):
    """
    Create a mesh object
    verts : (N, 3) array of vertex positions
    faces : (M, 3) array of triangle indices
    vertexcolor : (N, 3) or (3,) array of vertex colors
    """
    if type(vertexcolor) == str:
        vertexcolor = extract_rgb_values([vertexcolor])[0]
        
    if type(vertexcolor) == list or type(vertexcolor) == tuple:
        vertexcolor = np.array(vertexcolor)
        if vertexcolor.ndim == 1:
            vertexcolor = vertexcolor[None]
            vertexcolor = np.tile(vertexcolor, (verts.shape[0], 1))

    # create a smpl mesh
    mesh = go.Mesh3d(x=verts[:,0],
                    y=verts[:,1],
                    z=verts[:,2],
                    vertexcolor=vertexcolor,
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    lighting=Lighting()._asdict(), name=name,
                    opacity=opacity
                    )
    
    return mesh

def draw_3d_bbox(min_xyz, max_xyz, color='green', name=None):
    def get_3D_bbox_points(min_xyz,max_xyz):
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
        bbox_lines = np.array([[0,1],
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
        
        return bbox_points, bbox_lines
    # Generate lines for the 12 edges of the bounding box
    bbox_lines = []
    points, edges = get_3D_bbox_points(min_xyz, max_xyz)
    for i in range(12):
        s,e = edges[i]
        x_s,y_s,z_s = points[s]
        x_e,y_e,z_e = points[e]
        if name is None:
            name = ''
        bbox_lines.append(go.Scatter3d(
            x=[x_s, x_e],
            y=[y_s, y_e],
            z=[z_s, z_e],
            mode='lines',
            line=dict(color=color),name=name+f'trace_{i}'

        ))
    
    return bbox_lines
def draw_3d_line(start, end, color='red', name=None):
    if isinstance(start, list):
        start = np.array(start)
    if isinstance(end, list):
        end = np.array(end)
    if isinstance(start, torch.Tensor):
        start = start.cpu().numpy()
    if isinstance(end, torch.Tensor):
        end = end.cpu().numpy()
    assert start.shape == (3,)
    assert end.shape == (3,)
    x_s, y_s, z_s = start
    x_e, y_e, z_e = end

    line = go.Scatter3d(
        x=[x_s, x_e],
        y=[y_s, y_e],
        z=[z_s, z_e],
        mode='lines',
        line=dict(color=color, width=7),
        opacity=1.0,
        name=name
    )
    return line

def draw_3d_point(position, color='red', name=None):
    if isinstance(position, list):
        position = np.array(position)
    if isinstance(position, torch.Tensor):
        position = position.cpu().numpy()
    assert position.shape == (3,)
    x, y, z = position
    point = go.Scatter3d(
        x=[x],
        y=[y],
        z=[z],
        mode='markers',
        marker=dict(color=color, size=8),
        name=name
    )
    return point

def draw_3d_point_cloud(position, color='red', name=None):
    if isinstance(position, list):
        position = np.array(position)
    if isinstance(position, torch.Tensor):
        position = position.cpu().numpy()
    
    x, y, z = position[:,0], position[:,1], position[:,2]
    pc = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(color=color, size=2),
        name=name
    )
    return pc

def create_3d_figure(data:list, draw_grid=True):
    fig = go.Figure(data=data)
    
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data',
        xaxis = dict(visible=draw_grid),
        yaxis = dict(visible=draw_grid),
        zaxis =dict(visible=draw_grid)))
    return fig

def draw_image_plane_mesh_grid(image_plane_mesh_grid_coords:np.ndarray, dense=False):
    H, W = image_plane_mesh_grid_coords.shape[:2]
    edges = []
    if dense:
        for i in range(H):
            r = go.Scatter3d(
                x=image_plane_mesh_grid_coords[i, :, 0],
                y=image_plane_mesh_grid_coords[i, :, 1],
                z=image_plane_mesh_grid_coords[i, :, 2],
                mode='lines',
                line=dict(color='black', width=3),
                showlegend=True
            )
            edges.append(r)

        for i in range(W):
            c = go.Scatter3d(
                x=image_plane_mesh_grid_coords[:, i, 0],
                y=image_plane_mesh_grid_coords[:, i, 1],
                z=image_plane_mesh_grid_coords[:, i, 2],
                mode='lines',
                line=dict(color='black', width=3),
                showlegend=True
            )
            edges.append(c)
    else:
        # Calculate 4 corners of the image plane
        corners = np.array([image_plane_mesh_grid_coords[0,0], \
                            image_plane_mesh_grid_coords[H-1,0], \
                            image_plane_mesh_grid_coords[H-1,W-1], \
                            image_plane_mesh_grid_coords[0,W-1]
                            ]) # (4, 3)
        faces = np.array([[0,1,2],[0,2,3]])
        # Create the surface
        
        surface = go.Mesh3d(
            x=corners[:,0],
            y=corners[:,1],
            z=corners[:,2],
            i=faces[:,0],
            j=faces[:,1],
            k=faces[:,2],
            color='black',
            opacity=0.7
        )
        edges.append(surface)

    return edges

def draw_rays(ray_o, rays_d, name=None, color='black'):
    """
    ray_o: (3,) array of ray origins
    rays_d: (N, 3) array of ray directions
    """
    if isinstance(ray_o, torch.Tensor):
        ray_o = ray_o.cpu().numpy()
    if isinstance(rays_d, torch.Tensor):
        rays_d = rays_d.cpu().numpy()
    rays_d = rays_d.reshape(-1,3)
    # expand ray_o shape to rays_d shape
    rays_o = np.tile(ray_o[None], (len(rays_d), 1))
    rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True)
    rays = rays_o + rays_d
    scaled_rays = (rays * 100)
    rays_o = rays_o[::10000]
    scaled_rays = scaled_rays[::10000]
    
    x = [rays_o[:,0], scaled_rays[:,0]]
    y = [rays_o[:,0], scaled_rays[:,1]]
    z = [rays_o[:,0], scaled_rays[:,2]]
    line = go.Scatter3d(x=x, y=y, z=z, mode='lines', \
                        # marker=dict(color=color, size=10), name=name)
                        line=dict(color=color, width=1), name=name)
    return line

### 2D Rendering utils

def project_3D_joints(joints, K):
    # joints (24,3) is 3D joints in smpl coordinate
    # K (3,3)
    # project 3D joints to 2D image plane
    joints = joints 
    joints_img = np.matmul(K, joints.T).T
    joints_img = joints_img[:,:2] / joints_img[:,[2]]
    
    return joints_img

def draw_2D_joints(img, joints, point_size=1):
    # joints (24,3)
    # circle color is rainbow
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(joints) + 4)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    vis = img.copy()
    for i in range(len(joints)):
        cv2.circle(vis, (int(joints[i,0]), int(joints[i,1])), point_size, colors[i], -1)
    return vis

def get_3D_bbox_points(joints, bbox_offset=0.4):
    # joints: (24,3) in camera coordinate
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

def draw_3D_bbox_img(img, bbox_points, bbox_lines, intrinsic):
    # draw 3D bbox lines with opencv
    # bbox_points: (8,3) 3D points in camera coordinate
    # bbox_lines: (12,2)
    vis = img.copy()
    
    # 3D bbox points to 2D image plane
    K = intrinsic
    
    bbox_points = np.matmul(K, bbox_points.T).T
    
    bbox_points = bbox_points[:,:2] / bbox_points[:,[2]]
    
    edge_colors = [(255, 0, 0), (0, 255, 255), (0, 255, 0), (255, 0, 255), (0, 0, 255), (255, 255, 0), (255, 128, 0), (0, 128, 255), (128, 0, 255), (255, 255, 128), (255, 0, 128), (128, 255, 255)]
    for i in range(len(bbox_lines)):
        pt1 = (int(bbox_points[bbox_lines[i,0],0]), int(bbox_points[bbox_lines[i,0],1]))
        pt2 = (int(bbox_points[bbox_lines[i,1],0]), int(bbox_points[bbox_lines[i,1],1]))
        cv2.line(vis, pt1, pt2, edge_colors[i], 3)
    
    return vis, bbox_points

def draw_camera_coordinate(R, T):
    # R (3,3) rotation matrix, numpy.ndarray
    # T (3,) translation vector, numpy.ndarray
    
    camera_origin = np.matmul(R.T,-T)

    cam_origin_obj = draw_3d_point(position=camera_origin)
    x_axis = draw_3d_line(start=camera_origin, end=camera_origin + R[:,0], color='red')
    y_axis = draw_3d_line(start=camera_origin, end=camera_origin + R[:,1], color='green')
    z_axis = draw_3d_line(start=camera_origin, end=camera_origin + R[:,2], color='blue')
    return [cam_origin_obj, x_axis, y_axis, z_axis]

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,
    OrthographicCameras,
    PointLights, 
    RasterizationSettings, 
    MeshRendererWithFragments,
    MeshRasterizer,  
    SoftPhongShader,
)
from pytorch3d.renderer.mesh.textures import TexturesVertex

def set_pytorch3d_renderer(device='cpu'):
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. 
    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # Place a point light in front of the person. 
    lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            lights=lights
        )
    )

    return renderer, lights

def set_pytorch3d_camera(R, T, K, image_size=512, device='cpu'):
    # Rotation transfrom R (N,3,3)
    # Translation transform T (N,3)
    # Intrinsic matrix K (N,3,3)
    # R and T is "rotation and translation" in extrinsic matrix

    image_size_tensor = torch.tensor([[image_size, image_size]]).float().to(device)
    focal_length = torch.cat([K[:,0,0].view(-1,1), K[:,1,1].view(-1,1)], dim=-1).view(-1,2)
    princpt = torch.tensor([[K[:,0,2], K[:,1,2]]]).float().to(device).view(-1,2)
    cameras = PerspectiveCameras(device=device, focal_length=focal_length, principal_point=princpt, R=R, T=T, in_ndc=False, image_size=image_size_tensor)

    return cameras

def set_pytorch3d_camera_opengl(R,T,focal_len=5000, image_size=512, device='cpu'):

    # znear, zfar, aspect_ratio values are defined for THuman2.0 Dataset pre-precessed by DIFU
    znear = 0.1
    zfar = 40
    aspect_ratio = 1.0

    fovy = 2 * torch.atan(image_size / (2 * focal_len)) * 180 / torch.pi

    cameras = FoVPerspectiveCameras(znear=znear, zfar=zfar, aspect_ratio=aspect_ratio, fov=fovy, R=R, T=T, device=device)
    
    return cameras

import pdb

def render_mesh(renderer, lights, verts, faces, R=None, T=None, K=None, img=None, image_size=512, white_bg=False, opengl_camera=False, device='cpu'):
    # This method is to render a single mesh with pytorch3d
    # Later we will use this method to render multiple meshes
    
    if R is None:
        R = torch.eye(3).unsqueeze(0).to(device).float()
    if T is None:
        T = torch.zeros(1,3).to(device).float()
    if K is None:
        K = torch.eye(3).unsqueeze(0).to(device).float()
    
    cameras = set_pytorch3d_camera(R, T, K, image_size, device)
    if opengl_camera:
        cameras = set_pytorch3d_camera_opengl(R, T, image_size=image_size, device=device)
    verts = verts.to(device).view(1,-1,3)
    faces = faces.to(device).view(1,-1,3)
    verts_colors = (torch.zeros_like(verts)[0].cpu() + torch.tensor([0.5,0.7,0.7])[None]).to(device).float().unsqueeze(0)
    mesh_texture = TexturesVertex(verts_features=verts_colors)

    meshes = Meshes(verts, faces, textures=mesh_texture)
    renders, frags = renderer(meshes, lights=lights, cameras=cameras)
    
    # Get segmentation mask of the mesh
    depthmap = frags.zbuf.cpu().numpy() # (N,H,W,faces_per_pixel)
    depthmap[depthmap < 0] = 0.0
    segment = (depthmap > 0).astype(np.uint8)
    
    renders = renders.cpu().numpy()*255.0
    out = renders[0,...,:3].astype(np.uint8)
    if not img is None:
        img = img[...,:3][None]
        img = img * (1-segment) + renders[...,:3] * segment # image background
        out = img[0].astype(np.uint8)
    # else:
    #     renders = renders * segment # black background
    #     img = renders[...,:3]
    #     out = img[0].astype(np.uint8)
    
    return out

def heatmap_to_jet(heatmap_tensor):
    heatmap_tensor = heatmap_tensor.sum(0)
    
    heatmap = heatmap_tensor * 255
    heatmap = heatmap.astype(np.uint8)
    
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:,:,::-1]
    
    assert heatmap.shape[-1] == 3

    return heatmap

def opencv_image_blending(img1, img2, alpha=0.5):
    # img1, img2: (H,W,3) numpy
    # img1: background image
    # img2: foreground image
    # img2 is overlayed on img1
    # img2 is blended with img1
    # img2 is alpha blended with

    result = cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)

    return result

def vis_torchfied_image_data(img_torch, show=True, savename=None):
    # img_torch: (3,H,W) tensor
    # img: (H,W,3) numpy
    if img_torch.ndim == 4:
        if img_torch.shape[0] == 1:
            img_torch = img_torch.squeeze(0)
        else:
            raise ValueError('img_torch should be (3,H,W) or (H,W,3) tensor or (1,3,H,W) tensor')
    img = img_torch.detach().cpu().numpy().transpose(1,2,0)
    img = img * 255.0
    img = img.astype(np.uint8)
    if savename is not None:
        if '.png' not in savename:
            savename = savename + '.png'
        Image.fromarray(img).save(savename)