import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
from external.renderer.render_util import Pytorch3dRasterizer


def load_calib(calib_path):
    calib_data = np.loadtxt(calib_path, dtype=float)
    extrinsic = calib_data[:4, :4]
    intrinsic = calib_data[4:8, :4]
    calib_mat = np.matmul(intrinsic, extrinsic)
    calib_mat = torch.from_numpy(calib_mat).float()
    return calib_mat


def projection(points, calib, format='numpy'):
    """
    [fx   0  px][r11  r12  r13  t1]   [{fx*r11 + px*r31}  {fx*r12 + px*r32}  {fx*r13 + px*r33}  {fx*t1 + px*t3}]
    [ 0  fy  py][r21  r22  r23  t2] = [{fy*r21 + py*r31}  {fy*r22 + py*r32}  {fy*r23 + py*r33}  {fy*t2 + py*t2}]
    [ 0   0   1][r31  r32  r33  t3]   [             r31                r32                r33                t3]

      [P11 P12 P13 P14]   [P11 P12 P13] | [P14]
    = [P21 P22 P23 P24]   [P21 P22 P23] | [P24]
      [P31 P32 P33 P34] , [P31 P32 P33] | [P34]
    """
    if format == 'tensor':
        return torch.mm(calib[:3, :3], points.T).T + calib[:3, 3]
    else:
        return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]


def get_visibility(xy, z, faces):
    """get the visibility of vertices

    Args:
        xy (torch.tensor): [N,2]
        z (torch.tensor): [N,1]
        faces (torch.tensor): [N,3]
        size (int): resolution of rendered image
    """

    xyz = torch.cat((xy, -z), dim=1)
    xyz = (xyz + 1.0) / 2.0
    faces = faces.long()

    rasterizer = Pytorch3dRasterizer(image_size=2**12)
    meshes_screen = Meshes(verts=xyz[None, ...], faces=faces[None, ...])
    raster_settings = rasterizer.raster_settings

    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=raster_settings.image_size,
        blur_radius=raster_settings.blur_radius,
        faces_per_pixel=raster_settings.faces_per_pixel,
        bin_size=raster_settings.bin_size,
        max_faces_per_bin=raster_settings.max_faces_per_bin,
        perspective_correct=raster_settings.perspective_correct,
        cull_backfaces=raster_settings.cull_backfaces,
    )

    vis_vertices_id = torch.unique(faces[torch.unique(pix_to_face), :])
    vis_mask = torch.zeros(size=(z.shape[0], 1))
    vis_mask[vis_vertices_id] = 1.0

    # print("------------------------\n")
    # print(f"keep points : {vis_mask.sum()/len(vis_mask)}")

    return vis_mask

