import numpy as np
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights, 
    RasterizationSettings, 
    MeshRendererWithFragments,
    MeshRasterizer,  
    SoftPhongShader,
)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.mesh.textures import TexturesVertex, TexturesUV
from core.utils.body_util import load_obj

def _set_pytorch3d_renderer(cameras, device='cuda'):
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. 
    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # Place a point light in front of the person. 
    lights = PointLights(device=device, location=[[0.0, 1.0, 1.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)) # black background
        )
    )

    return renderer, lights

def set_pytorch3d_camera_object(image_size, R, T, K, device='cuda'):
    '''
    image_size: (H, W), tuple
    R: (N, 3, 3) rotation matrix, np.ndarray or torch.tensor
    T: (N, 3) translation vector, np.ndarray or torch.tensor
    K: (N, 3, 3) camera intrinsic matrix, np.ndarray or torch.tensor
    '''

    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).float().to(device)
    if isinstance(T, np.ndarray):
        T = torch.from_numpy(T).float().to(device)
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float().to(device)
    if R.dim() == 2:
        R = R.unsqueeze(0)
    if T.dim() == 1:
        T = T.unsqueeze(0)
    if K.dim() == 2:
        K = K.unsqueeze(0)

    H = image_size[0]
    W = image_size[1]

    image_size_tensor = torch.tensor((H, W), dtype=torch.float32).repeat(5,1)
    focal_length = torch.cat([K[:,0,0].view(-1,1), K[:,1,1].view(-1,1)], dim=-1).view(-1,2).repeat(5,1)
    focal_length = -1 * focal_length # In pytorch3D rendering system, focal length should be negated.
    princpt = torch.tensor([[K[:,0,2], K[:,1,2]]]).float().to(device).view(-1,2).repeat(5,1)
    cameras = PerspectiveCameras(device=device, 
                             focal_length=focal_length, 
                             principal_point=princpt, 
                             R=R, T=T, 
                             in_ndc=False, 
                             image_size=image_size_tensor)
    
    return cameras

def set_pytorch3d_renderer(image_size, R, T, K, device):
    # usage example : rendering = rndr(mesh, R=R, T=T, lights=ligths)

    cameras = set_pytorch3d_camera_object(image_size, R, T, K, device)
    rndr, lights = _set_pytorch3d_renderer(cameras, device=device)
    
    return rndr

def set_texture_map_pytorch3d(texture_map, faces, uv_coordinate, device='cuda', normalize=False):
    # TODO: Implement this function
    '''
    texture: (H,W,3) np.ndarray [0,1]
    '''
    
    texture_map = texture_map.unsqueeze(0) # (1,H,W,C)
    verts_uv = uv_coordinate
    texture = TexturesUV(maps=texture_map, faces_uvs=faces[None], verts_uvs=verts_uv)
    
    return texture

def set_textured_SMPL_mesh_pytorch3d(verts, faces, texture_map, uv_coords, normalize_texture_map_tensor=False, device='cuda'):
    if isinstance(verts, np.ndarray):
        verts = torch.from_numpy(verts).float().to(device)
    if verts.dim() == 2:
        verts = verts.unsqueeze(0)
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces).long().to(device)
    if faces.dim() == 2:
        faces = faces.unsqueeze(0)
    
    if isinstance(texture_map, np.ndarray):
        texture_map = torch.from_numpy(np.array(texture_map)).float()
    if texture_map.dim() == 3:
        texture_map = texture_map.unsqueeze(0)
    
    if isinstance(uv_coords, np.ndarray):
        uv_coords = torch.from_numpy(uv_coords).float().to(device)
    if uv_coords.dim() == 2:
        uv_coords = uv_coords.unsqueeze(0)

    if normalize_texture_map_tensor:
        texture_map = texture_map / 255.0
    texture_uv = set_texture_map_pytorch3d(texture_map, faces, texture_map.verts_uvs_list(), device=device)
    
    mesh = Meshes(
        verts=verts,
        faces=faces,
        textures=texture_uv
    )

    return mesh

class Pytorch3D_Renderer(nn.Module):
    def __init__(self, rendering_image_size, R, T, K, texture_map_size, device='cuda'):
        super(Pytorch3D_Renderer, self).__init__()
        self.device = device
        self.rendering_image_size = rendering_image_size
        self.R = R
        self.T = T
        self.K = K
        self.renderer = set_pytorch3d_renderer(rendering_image_size, R, T, K, device)

        uv_obj_file_path = 'third_parties/smpl/models/smpl_uv.obj'
        uv_data = load_obj(uv_obj_file_path)
        self.uv_coords, self.smpl_faces = self.load_uv(uv_data, image_size=texture_map_size)
    
    def forward(self, verts, texture, R, T, normalize_texture_tensor=False):
        mesh = set_textured_SMPL_mesh_pytorch3d(verts, self.smpl_faces, texture, self.uv_coords, normalize_texture_map_tensor=normalize_texture_tensor, device=self.device)
        lights = self.create_camera_light(R, T)
        return self.renderer(mesh, R=R, T=T, lights=lights)
    
    def create_camera_light(self, R:torch.Tensor, T:torch.Tensor):
        '''
        R: (N, 3, 3) rotation matrix, torch.Tensor
        T: (N, 3) translation vector, torch.Tensor
        '''
        camera_center = -R.permute(0,2,1) @ T[:,:,None] # in numpy, camera_center = -R @ T[:,None]
        camera_center = camera_center.squeeze(-1)
        return PointLights(device=self.device, location=camera_center) # Input location shape must be (N, 3)

    def load_uv(self, data, image_size):
        uv = data['uv']
        uv = uv * (image_size - 1)
        uv = uv.astype(int)
        uv[:, 1] = (image_size - 1) - uv[:, 1]
        uv_mapping = torch.from_numpy(uv)
        faces_uv = torch.from_numpy(data['faces_uv'])
        vt_to_v = data['vt_to_v']
        vt_to_v_index = torch.tensor([vt_to_v[idx] for idx in range(7576)], requires_grad=False).long()
        
        return uv_mapping, faces_uv
        
        