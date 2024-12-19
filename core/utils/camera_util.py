import numpy as np
import torch
import cv2

def _update_extrinsics(
        extrinsics, 
        angle, 
        trans=None, 
        rotate_axis='y'):
    r""" Uptate camera extrinsics when rotating it around a standard axis.

    Args:
        - extrinsics: Array (3, 3)
        - angle: Float
        - trans: Array (3, )
        - rotate_axis: String

    Returns:
        - Array (3, 3)
    """
    E = extrinsics
    inv_E = np.linalg.inv(E)

    camrot = inv_E[:3, :3] # Camera rotation and position
    campos = inv_E[:3, 3] # Camera position in the world
    if trans is not None:
        campos -= trans # A vector SMPL root camera center

    rot_y_axis = camrot.T[1, 1]
    if rot_y_axis < 0.:
        angle = -angle
    
    rotate_coord = {
        'x': 0, 'y': 1, 'z':2
    }
    grot_vec = np.array([0., 0., 0.])
    grot_vec[rotate_coord[rotate_axis]] = angle
    grot_mtx = cv2.Rodrigues(grot_vec)[0].astype('float32')

    rot_campos = grot_mtx.dot(campos) 
    rot_camrot = grot_mtx.dot(camrot)
    if trans is not None:
        rot_campos += trans # camera position to world coordinate
    
    new_E = np.identity(4)
    new_E[:3, :3] = rot_camrot.T
    new_E[:3, 3] = -rot_camrot.T.dot(rot_campos)

    return new_E

    
def get_camrot(campos, lookat=None, inv_camera=False):
    r""" Compute rotation part of extrinsic matrix from camera posistion and
         where it looks at.

    Args:
        - campos: Array (3, )
        - lookat: Array (3, )
        - inv_camera: Boolean

    Returns:
        - Array (3, 3)

    Reference: http://ksimek.github.io/2012/08/22/extrinsic/
    """

    if lookat is None:
        lookat = np.array([0., 0., 0.], dtype=np.float32)

    # define up, forward, and right vectors
    up = np.array([0., 1., 0.], dtype=np.float32)
    if inv_camera:
        up[1] *= -1.0
    forward = lookat - campos
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    camrot = np.array([right, up, forward], dtype=np.float32)
    return camrot


def rotate_camera_by_frame_idx(
        extrinsics, 
        frame_idx, 
        trans=None,
        rotate_axis='y',
        period=196,
        inv_angle=False):
    r""" Get camera extrinsics based on frame index and rotation period.

    Args:
        - extrinsics: Array (3, 3)
        - frame_idx: Integer
        - trans: Array (3, )
        - rotate_axis: String
        - period: Integer
        - inv_angle: Boolean (clockwise/counterclockwise)

    Returns:
        - Array (3, 3)
    """

    angle = 2 * np.pi * (frame_idx / period)
    if inv_angle:
        angle = -angle
    return _update_extrinsics(
                extrinsics, angle, trans, rotate_axis)

def rotate_camera_by_angle(
        extrinsics, 
        angle,
        trans=None,
        rotate_axis='y',
        inv_angle=False):
    r""" Get camera extrinsics based on frame index and rotation period.

    Args:
        - extrinsics: Array (3, 3)
        - angle: Float (1, ) (degree)
        - trans: Array (3, )
        - rotate_axis: String
        - inv_angle: Boolean (clockwise/counterclockwise)

    Returns:
        - Array (3, 3)
    """

    angle = np.deg2rad(angle)
    if inv_angle:
        angle = -angle
    return _update_extrinsics(
                extrinsics, angle, trans, rotate_axis)

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
    return E.dot(np.linalg.inv(global_tfms)).astype(np.float32)


def get_rays_from_KRT(H, W, K, R, T):
    r""" Sample rays on an image based on camera matrices (K, R and T)

    Args:
        - H: Integer
        - W: Integer
        - K: Array (3, 3)
        - R: Array (3, 3)
        - T: Array (3, )
        
    Returns:
        - rays_o: Array (H, W, 3)
        - rays_d: Array (H, W, 3)
    """

    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o.astype(np.float32), rays_d.astype(np.float32)

def _get_samples_along_ray(N_rays, near, far, n_samples):
    t_vals = torch.linspace(0., 1., steps=n_samples).to(near) # Proportion of locations
    z_vals = near * (1.-t_vals) + far * (t_vals) # mapping proportion line to actual length space across batched rays
    return z_vals.expand([N_rays, n_samples]) 


def _stratified_sampling(z_vals):
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    upper = torch.cat([mids, z_vals[...,-1:]], -1)
    lower = torch.cat([z_vals[...,:1], mids], -1)
    
    t_rand = torch.rand(z_vals.shape).to(z_vals)
    z_vals = lower + (upper - lower) * t_rand

    return z_vals

def sample_points_along_ray(rays_o, rays_d, near, far, n_samples, perturbation, **__):
    N_rays = rays_o.shape[0]

    z_vals = _get_samples_along_ray(N_rays, near, far, n_samples)

    if perturbation > 0:
        z_vals = _stratified_sampling(z_vals=z_vals)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

    return z_vals, pts

def rays_intersect_3d_bbox(bounds, ray_o, ray_d):
    r"""calculate intersections with 3d bounding box
        Args:
            - bounds: dictionary or list
            - ray_o: (N_rays, 3)
            - ray_d, (N_rays, 3)
        Output:
            - near: (N_VALID_RAYS, )
            - far: (N_VALID_RAYS, )
            - mask_at_box: (N_RAYS, )
    """

    if isinstance(bounds, dict):
        bounds = np.stack([bounds['min_xyz'], bounds['max_xyz']], axis=0)
    assert bounds.shape == (2,3)
    
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    # bounds = np.array([[-1.0, -2.0, -1.0], [1.0, 2.0, 1.0]])
    
    nominator = bounds[None] - ray_o[:, None] # (N_rays, 2, 3)
    # calculate the step of intersections at six planes of the 3d bounding box
    ray_d[np.abs(ray_d) < 1e-5] = 1e-5
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6) # (N_rays, 6)
    # calculate the six intersections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None] # (N_rays, 6, 3)
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))  # (N_rays, 6)
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2  #(N_rays, ) # Which rays intersect the box twice
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3) # (N_VALID_rays, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)
    return near, far, mask_at_box

def _get_patch_ray_indices(
            ray_mask, 
            candidate_mask, 
            patch_size, 
            H, W, candidate_is_subject=True):

    assert len(ray_mask.shape) == 1
    assert ray_mask.dtype == np.bool_
    assert candidate_mask.dtype == np.bool_

    valid_ys, valid_xs = np.where(candidate_mask)
    
    # determine patch center
    select_idx = np.random.choice(valid_ys.shape[0], 
                                    size=[1], replace=False)[0]
    center_x = valid_xs[select_idx]
    center_y = valid_ys[select_idx]

    # determine patch boundary
    half_patch_size = patch_size // 2
    x_min = np.clip(a=center_x-half_patch_size, 
                    a_min=0, 
                    a_max=W-patch_size)
    x_max = x_min + patch_size
    y_min = np.clip(a=center_y-half_patch_size,
                    a_min=0,
                    a_max=H-patch_size)
    y_max = y_min + patch_size
    
    sel_ray_mask = np.zeros_like(candidate_mask)
    sel_ray_mask[y_min:y_max, x_min:x_max] = True

    #####################################################
    ## Below we determine the selected ray indices
    ## and patch valid mask

    sel_ray_mask = sel_ray_mask.reshape(-1)
    inter_mask = np.bitwise_and(sel_ray_mask, ray_mask) # the mask map has squares in the valid region
    select_masked_inds = np.where(inter_mask) # indices of pixels
    
    masked_indices = np.cumsum(ray_mask) - 1 # indices of valid rays
    select_inds = masked_indices[select_masked_inds]
    
    inter_mask = inter_mask.reshape(H, W)
    inter_mask_candidate_area = np.bitwise_and(inter_mask, candidate_mask) if candidate_is_subject else np.bitwise_and(inter_mask, np.bitwise_not(candidate_mask))

    return select_inds, \
            inter_mask[y_min:y_max, x_min:x_max], \
            np.array([x_min, y_min]), np.array([x_max, y_max]), \
            inter_mask_candidate_area[y_min:y_max, x_min:x_max]

def get_patch_ray_indices(
            sample_subject_ratio, 
            N_patch, 
            ray_mask, # flattened single channel image, each element is validity of ray for corresponding pixel
            subject_mask, # Segmentation map
            bbox_mask, # Bounding box region segmentation
            patch_size, 
            H, W):
    assert subject_mask.dtype == np.bool_
    assert bbox_mask.dtype == np.bool_

    bbox_exclude_subject_mask = np.bitwise_and(
        bbox_mask,
        np.bitwise_not(subject_mask)
    )
    list_ray_indices = []
    list_mask = []
    list_xy_min = []
    list_xy_max = []
    list_alpha_gt = []
    
    total_rays = 0
    patch_div_indices = [total_rays]
    for i in range(N_patch):
        # let p = cfg.patch.sample_subject_ratio
        # prob p: we sample on subject area
        # prob (1-p): we sample on non-subject area but still in bbox
        
        if np.random.rand(1)[0] < sample_subject_ratio:
            candidate_mask = subject_mask
            subject = True
        else:
            candidate_mask = bbox_exclude_subject_mask
            subject = False

        ray_indices, mask, xy_min, xy_max, patch_alpha_gt = \
            _get_patch_ray_indices(ray_mask, candidate_mask, 
                                        patch_size, H, W, candidate_is_subject=subject)

        assert len(ray_indices.shape) == 1
        total_rays += len(ray_indices)
        list_ray_indices.append(ray_indices)
        list_mask.append(mask)
        list_xy_min.append(xy_min)
        list_xy_max.append(xy_max)
        list_alpha_gt.append(patch_alpha_gt)
        
        patch_div_indices.append(total_rays)
    
    select_inds = np.concatenate(list_ray_indices, axis=0)
    patch_info = {
        'mask': np.stack(list_mask, axis=0),
        'alpha_gt' : np.stack(list_alpha_gt, axis=0),
        'xy_min': np.stack(list_xy_min, axis=0),
        'xy_max': np.stack(list_xy_max, axis=0)
    }
    patch_div_indices = np.array(patch_div_indices)
    
    return select_inds, patch_info, patch_div_indices

def select_rays(select_inds, rays_o, rays_d, ray_img, near, far):
    rays_o = rays_o[select_inds]
    rays_d = rays_d[select_inds]
    ray_img = ray_img[select_inds]
    near = near[select_inds]
    far = far[select_inds]
    return rays_o, rays_d, ray_img, near, far

def sample_patch_rays(N_patches, patch_size, sample_subject_ratio, img, H, W,
                          subject_mask, bbox_mask, ray_mask,
                          rays_o, rays_d, ray_img, near, far):
        
    select_inds, patch_info, patch_div_indices = \
        get_patch_ray_indices(
            sample_subject_ratio=sample_subject_ratio,
            N_patch=N_patches, 
            ray_mask=ray_mask, 
            subject_mask=subject_mask, 
            bbox_mask=bbox_mask,
            patch_size=patch_size, 
            H=H, W=W)
    
    rays_o, rays_d, ray_img, near, far = select_rays(
        select_inds, rays_o, rays_d, ray_img, near, far)
    
    targets = []
    
    for i in range(N_patches):
        x_min, y_min = patch_info['xy_min'][i] 
        x_max, y_max = patch_info['xy_max'][i]
        targets.append(img[y_min:y_max, x_min:x_max])
    target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)

    patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)
    patch_alpha_gt = patch_info['alpha_gt'] # boolean array (N_patches, P, P)

    return rays_o, rays_d, ray_img, near, far, \
            target_patches, patch_masks, patch_div_indices, patch_alpha_gt

def unpack_patch_imgs_to_img_shape(rgbs, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

    return patch_imgs

def construct_ray_data(img, alpha, alpha_1ch,
                               H,
                               W,
                               skel_info:dict,
                               camera_dict:dict,
                               resize_img_scale=1.0,
                               ray_mode = 'image',
                               keyfilter={},
                               N_patches=-1,
                               patch_size=-1,
                               sample_subject_ratio=-1,**_
                               ):
    """
    Return : 
        If ray_mode == 'image':
            return {batch_rays, rays_o, rays_d, ray_img, near, far}
        
        If ray_mode == 'patch':
            return {batch_rays, rays_o, rays_d, ray_img, near, far,
                target_patches, patch_masks, patch_div_indices, mask_raw}
    """
    if ray_mode == 'patch':
        assert N_patches > 0 and patch_size > 0 and sample_subject_ratio > 0

    dst_bbox = skel_info['bbox'] # dict, contains {minxyz, maxxyz} got from 3D skeleton
    
    K = camera_dict['intrinsics'][:3, :3].copy().astype(np.float32)
    K[:2] *= resize_img_scale

    E_data = camera_dict['extrinsics'].copy().astype(np.float32)
    E = apply_global_tfm_to_camera(
            E=E_data,
            Rh=skel_info['Rh'],
            Th=skel_info['Th']).astype(np.float32)
    R = E[:3, :3]
    T = E[:3, 3]
    rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
    ray_img = img.reshape(-1, 3) 
    rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
    rays_d = rays_d.reshape(-1, 3)

    near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
    x,y,w,h = cv2.boundingRect(ray_mask.reshape(H,W).astype(np.uint8))
    tight_ray_mask = [x,y,w,h]
    rays_o = rays_o[ray_mask]
    rays_d = rays_d[ray_mask]
    ray_img = ray_img[ray_mask]
    ray_alpha = alpha_1ch.reshape(-1)[ray_mask]
    near = near[:, None].astype('float32')
    far = far[:, None].astype('float32')

    if ray_mode == 'patch':
        rays_o, rays_d, ray_img, near, far, \
        target_rgb_patches, patch_masks, patch_div_indices, target_alpha_patches = \
            sample_patch_rays(N_patches=N_patches, patch_size=patch_size, 
                                sample_subject_ratio=sample_subject_ratio,
                                img=img, H=H, W=W,
                                subject_mask=alpha[:, :, 0] > 0.,
                                bbox_mask=ray_mask.reshape(H, W),
                                ray_mask=ray_mask,
                                rays_o=rays_o, 
                                rays_d=rays_d, 
                                ray_img=ray_img, 
                                near=near, 
                                far=far)
    elif ray_mode == 'image':
        pass
    else:
        assert False, f"Ivalid Ray Shoot Mode: {ray_mode}"

    batch_rays = np.stack([rays_o, rays_d], axis=0)
    
    out = {}
    if 'rays' in keyfilter:
        out.update({
            'extrinsics':E,
            'intrinsics':K,
            'ray_mask': ray_mask,
            'tight_ray_mask': tight_ray_mask,
            'rays': batch_rays,
            'near': near,
            'far': far,
            'target_rgbs': ray_img,
            'target_alpha': ray_alpha
            })

        if ray_mode == 'patch':
            out.update({
                'patch_div_indices': patch_div_indices,
                'patch_masks': patch_masks,
                'target_rgb_patches': target_rgb_patches,
                'target_alpha_patches': target_alpha_patches
                })

    # if 'target_rgbs' in keyfilter:
    #     out['target_rgbs'] = ray_img
    return out