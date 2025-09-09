from pytorch3d import ops
import torch

def get_bbox_from_smpl(vs, factor=1.2):
    assert vs.shape[0] == 1
    min_vert = vs.min(dim=1).values
    max_vert = vs.max(dim=1).values

    c = (max_vert + min_vert) / 2
    s = (max_vert - min_vert) / 2
    s = s.max(dim=-1).values * factor

    min_vert = c - s[:, None]
    max_vert = c + s[:, None]
    return torch.cat([min_vert, max_vert], dim=0)

class SMPLDeformer():
    def __init__(self, smpl_model) -> None:
        self.body_model = smpl_model
        self.strategy = "nearest_neighbor"

    def initialize(self, betas, device):
        # define template canonical pose
        #   T-pose does't work very well for some cases (legs are too close) 
        # convert to canonical space(Big-Pose)
        self.batch_size = batch_size = betas.shape[0]
        body_pose_cnl = torch.zeros((batch_size, 69), device=device)
        body_pose_cnl[:, 2] = torch.pi / 4
        body_pose_cnl[:, 5] = -torch.pi / 4
        body_pose_cnl[:, 20] = -torch.pi / 6
        body_pose_cnl[:, 23] = torch.pi / 6
        smpl_outputs = self.body_model(betas=betas, body_pose=body_pose_cnl)
        vertices = smpl_outputs.vertices
        self.cnl_bbox = get_bbox_from_smpl(vertices[0:1].detach())
        self.cnl_bbox = self.cnl_bbox.to(device)
        
        self.cnl_bbox_center = (self.cnl_bbox[0, :3] + self.cnl_bbox[1, :3]) / 2
        self.cnl_bbox_scale = self.cnl_bbox[1, :3] - self.cnl_bbox[0, :3]
        self.cnl_bbox_min = self.cnl_bbox[0, :3]
        self.cnl_bbox_max = self.cnl_bbox[1, :3]

        self.cnl_vertices = vertices[0].detach()
        
        self.T_template = smpl_outputs.T
        self.vs_template = smpl_outputs.vertices
        self.pose_offset_cnl = smpl_outputs.pose_offsets
        self.shape_offset_cnl = smpl_outputs.shape_offsets

    def prepare_deformer(self, smpl_params_target, smpl_params_obs):
        device = smpl_params_target["betas"].device
        if next(self.body_model.parameters()).device != device:
            self.body_model = self.body_model.to(device)
            self.body_model.eval()

        self.initialize(smpl_params_target["betas"], device)
        
        global_orient = torch.zeros((1, 3), device=device).float()
        transl = torch.zeros((1, 3), device=device).float()
        
        smpl_outputs_target = self.body_model(betas=smpl_params_target["betas"],
                                    body_pose=smpl_params_target["body_pose"],
                                    global_orient=global_orient,
                                    transl=transl)
        T_inv_target_to_cnl, T_fwd_target, vertices_target = self.calculate_transform(smpl_outputs_target)
        self.vertices_target = vertices_target.float()
        self.T_inv_target_to_cnl = T_inv_target_to_cnl.float()
        self.T_fwd_target = T_fwd_target.float()
        
        smpl_outputs_obs = self.body_model(betas=smpl_params_obs["betas"],
                                    body_pose=smpl_params_obs["body_pose"],
                                    global_orient=global_orient,
                                    transl=transl)
        T_inv_obs_to_cnl, T_fwd_obs, vertices_obs = self.calculate_transform(smpl_outputs_obs)
        
        self.vertices_obs = vertices_obs.float()
        self.T_inv_obs_to_cnl = T_inv_obs_to_cnl.float()
        self.T_fwd_obs = T_fwd_obs.float()

        self.vertices_cnl = self.vs_template
        self.w2s = torch.eye(4, device=device).float()

    def calculate_transform(self, smpl_outputs):
        # remove & reapply the blendshape
        s2w = smpl_outputs.A[:, 0] # Identical matrix
        w2s = torch.inverse(s2w)
        T_inv = torch.inverse(smpl_outputs.T.float()).clone() @ s2w[:, None]
        T_inv[..., :3, 3] += self.pose_offset_cnl - smpl_outputs.pose_offsets
        T_inv[..., :3, 3] += self.shape_offset_cnl - smpl_outputs.shape_offsets
        T_inv = self.T_template @ T_inv
        
        T_fwd = torch.inverse(self.T_template.float()).clone() @ s2w[:, None]
        T_fwd[..., :3, 3] += smpl_outputs.pose_offsets - self.pose_offset_cnl
        T_fwd[..., :3, 3] += smpl_outputs.shape_offsets - self.shape_offset_cnl
        T_fwd = smpl_outputs.T.float().clone() @ T_fwd

        vertices = smpl_outputs.vertices

        return T_inv, T_fwd, vertices

    def transform_rays_w2s(self, rays):
        """transform rays from world to smpl coordinate system"""
        w2s = self.w2s
        rays.o = (rays.o @ w2s[:, :3, :3].permute(0, 2, 1)) + w2s[:, None, :3, 3]
        rays.d = (rays.d @ w2s[:, :3, :3].permute(0, 2, 1)).to(rays.d)
        d = torch.norm(rays.o, dim=-1)
        rays.near = d - 1
        rays.far = d + 1

    def closest_idx_in_obs(self, pts):
        # find nearest neighbors
        pts = pts.reshape(self.batch_size, -1, 3)
        ref_vertices = self.vertices_obs

        _, idx, _ = ops.knn_points(pts.float(), ref_vertices, K=1)

        idx = idx.squeeze(-1).reshape(-1)
        
        return idx
    
    def deform_to_obs(self, pts, closest_idx):
        """
        If 'forward_lbs' is True, transform canonical space to deformed space.
        -> pts must be in canonical space

        If 'forward_lbs' is False, transform deformed space to canonical space.
        -> pts must be in deformed space
        
        pts (torch.Tensor)(batch_size, #pts, 3) : 3D points in canonical space
        closest_idx (torch.Tensor) (batch_size, #pts) or (batch_size, #pts, 1): closest mesh vertex index in canonical space 
        """
        
        batch_size = self.vertices_target.shape[0]
        assert pts.shape[0] == batch_size or closest_idx.shape[0] == batch_size

        pts = pts.reshape(batch_size, -1, 3)
        
        if closest_idx.shape[-1] == 1:
            idx = closest_idx.squeeze(-1).view(batch_size,-1)
        else:
            idx = closest_idx.view(batch_size,-1)

        # T ~ (batch_size, #pts, #neighbors, 4, 4)
        pts_processed = torch.zeros_like(pts, dtype=torch.float32)
        for i in range(batch_size):
            Tv_obs = self.T_fwd_obs[i][idx[i]]
            T = Tv_obs
            pts_processed[i] = (T[..., :3, :3] @ pts[i][..., None]).squeeze(-1) + T[..., :3, 3]
            
        
        return pts_processed.reshape(-1, 3)
    
    def deform_to_cnl(self, pts, closest_idx):
        """
        If 'forward_lbs' is True, transform canonical space to deformed space.
        -> pts must be in canonical space

        If 'forward_lbs' is False, transform deformed space to canonical space.
        -> pts must be in deformed space

        pts (torch.Tensor)(batch_size, #pts, 3) : 3D points in target space
        closest_idx (torch.Tensor) (batch_size, #pts) or (batch_size, #pts, 1): closest mesh vertex index in target space 
        """
        
        batch_size = self.vertices_target.shape[0]
        assert pts.shape[0] == batch_size or closest_idx.shape[0] == batch_size

        pts = pts.reshape(batch_size, -1, 3)
        
        if closest_idx.shape[-1] == 1:
            idx = closest_idx.squeeze(-1)
        else:
            idx = closest_idx

        # T ~ (batch_size, #pts, #neighbors, 4, 4)
        pts_processed = torch.zeros_like(pts, dtype=torch.float32)
        for i in range(batch_size):
            Tv_inv = self.T_inv_target_to_cnl[i][idx[i]]
            T = Tv_inv
            pts_processed[i] = (T[..., :3, :3] @ pts[i][..., None]).squeeze(-1) + T[..., :3, 3]
        
        return pts_processed.reshape(-1, 3)
    