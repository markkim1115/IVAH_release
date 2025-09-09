import torch
import torch.nn as nn
import torch.nn.functional as F
from core.nets.human_nerf.deformers.smpl_deformer import SMPLDeformer
from core.nets.human_nerf.uv_generator import UV_Generator
from core.nets.human_nerf.component_factory import load_positional_embedder, load_canonical_mlp
from core.nets.human_nerf.smpl_feature_volume import SMPL_Feature_Volume
from core.utils.misc import count_parameters
from configs import cfg

from third_parties.smpl.smpl import load_smpl_model
from pytorch3d import ops

class HumanNeRF(nn.Module):
    def __init__(self):
        super(HumanNeRF, self).__init__()
        self.smpl_model = load_smpl_model()
        self.deformer = SMPLDeformer(smpl_model=self.smpl_model)
        
        self.lin_pix_aligned = nn.Sequential(nn.Linear(64+27, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))
        self.lin_invisible = nn.Sequential(nn.Linear(64+27, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))
        
        if cfg.use_uv_inpainter:
            uv_gen_inp_ch = 3
            self.UV_generator = UV_Generator(in_channels=uv_gen_inp_ch, uv_map_size=cfg.uv_map.uv_map_size, cfg=cfg)
        
        if cfg.use_smpl_3d_feature:
            self.smpl_feat_decoder = SMPL_Feature_Volume(smpl_model=self.smpl_model, in_ch=64+27)
            self.mlp_project_3d = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))
        
        # canonical positional encoding
        get_embedder = load_positional_embedder(cfg.embedder.module)
        cnl_pos_embed_fn, cnl_pos_embed_size = \
            get_embedder(cfg.canonical_mlp.multires, 
                         cfg.canonical_mlp.i_embed)
        self.pos_embed_fn = cnl_pos_embed_fn

        if cfg.append_rgb:
            get_embedder = load_positional_embedder(cfg.embedder.module)
            view_embed_fn, view_embed_size = get_embedder(4,0)
            self.view_embed_fn = view_embed_fn

        # Input feature dimension
        nerf_f_pixel_dim = 32
        input_ch = cnl_pos_embed_size + nerf_f_pixel_dim
        self.input_component_dims = {'cnl_pos_embed_size': cnl_pos_embed_size, 'f_pixel_dim': nerf_f_pixel_dim}
                
        self.cnl_mlp = load_canonical_mlp(cfg.canonical_mlp.module)(
                                                                    input_ch=input_ch, 
                                                                    mlp_depth=cfg.canonical_mlp.mlp_depth, 
                                                                    mlp_width=cfg.canonical_mlp.mlp_width,
                                                                    comps_dim=self.input_component_dims)
        
        print("Parameters of NeRF decoder")
        print('Number of parameters | NeRF decoder: {} (total, trainable)'.format(count_parameters(self.cnl_mlp)))
        
    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))

    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, bgcolor=None):
        def set_act_fn():
            if cfg.rgb_act_fn == 'widened_sigmoid':
                rgb_act_fn = lambda x: torch.sigmoid(x)*(1 + 2*0.001) - 0.001
            else:
                rgb_act_fn = torch.sigmoid
            if cfg.alpha_act_fn == 'shifted-softplus':
                density_act_fn = lambda x: F.softplus(x-1)
            else:
                density_act_fn = F.relu
            
            return rgb_act_fn, density_act_fn
        
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)
        
        rgb_act_fn, density_act_fn = set_act_fn()

        dists = z_vals[...,1:] - z_vals[...,:-1] # sample intervals: [N_rays, N_samples-1]

        infinity_dists = torch.Tensor([1e10]) 
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1) # move the start point to the each sample point?

        rgb = rgb_act_fn(raw[...,:3]) # [N_rays, N_samples, 3]
        
        alpha = _raw2alpha(raw[...,3], dists, act_fn=density_act_fn)  # [N_rays, N_samples]
        alpha = alpha * raw_mask[:, :, 0]

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.
        
        return rgb_map, acc_map, weights, depth_map, alpha

    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        return rays_o, rays_d, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near) # Proportion of locations
        z_vals = near * (1.-t_vals) + far * (t_vals) # mapping proportion line to actual length space across batched rays
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals
    
    
    def sample_features(self, sample_points:torch.Tensor, 
                        feature_data:torch.Tensor,
                        extrinsics:torch.Tensor, 
                        intrinsics:torch.Tensor,
                        ):
        '''
        Sample image features from feature map
        '''
        # sample_points: (n_points, 3)
        # E: (4, 4)
        # K: (3, 3)
        # Feature map: (1, C, H, W)
        # return f_pixel (1, n_points, C)
        H = feature_data.shape[-2]
        W = feature_data.shape[-1]
        size = torch.tensor([W, H]).to(feature_data).float()[None] # (1, 2)
        # size = torch.tensor(feature_data.shape[-2:]).to(feature_data).float()[None] # (1, 2)

        sample_points = sample_points.reshape(-1, 3)
        R = extrinsics[:3,:3]
        T = extrinsics[:3,[3]]
        
        uv = (intrinsics @ (torch.matmul(R, sample_points.permute(1,0)) + T)).permute(1,0) # n_points, 3
        uv = uv[:,:2] / uv[:,2:3]
        uv_norm = ((uv / size) * 2 - 1)[None,None] # 1, 1, n_points, 2
        
        f_pixel = F.grid_sample(feature_data, 
                      uv_norm, 
                      mode='bilinear', 
                      padding_mode='zeros', 
                      align_corners=True)
        
        return f_pixel.squeeze(0).permute(1,2,0), uv
    
    def _render_rays(
            self, 
            rays_o,
            rays_d,
            pts, 
            z_vals,
            bgcolor,
            pos_embed_fn,
            output_list = [],
            **kwargs):
        
        N_rays, N_samples, _ = pts.shape
        
        pts_flat = pts.reshape(-1, 3) # [N_rays x N_samples, 3]
        
        chunk = cfg.netchunk_per_gpu * 1
        
        deformer : SMPLDeformer = self.deformer
        dist_sq, closest_verts_idx_target, _ = ops.knn_points(pts_flat[None], deformer.vertices_target, K=1)
        closest_verts_idx_target = closest_verts_idx_target[...,0]

        valid_queries_mask = dist_sq < 0.05 ** 2
        valid_queries_mask = valid_queries_mask.view(-1)
        x_skel = deformer.deform_to_cnl(pts_flat, closest_verts_idx_target)
        x_skel_pruned = x_skel[valid_queries_mask]
        
        dist_sq, closest_verts_idx_cnl, _ = ops.knn_points(x_skel_pruned[None], deformer.vertices_cnl, K=1)
        x_obs_pruned = deformer.deform_to_obs(x_skel_pruned, closest_verts_idx_cnl) # [N_valid_samples, 3]

        closest_verts_index_ = closest_verts_idx_target.reshape(-1)

        ## Sample pixel aligned features from images
        obs_smpl_vis_mask = kwargs['inp_visible_vertices_mask']
        obs_view_visibility = obs_smpl_vis_mask[closest_verts_index_][valid_queries_mask][None]
        
        rgb_obs, _ = self.sample_features(x_obs_pruned, kwargs['inp_img_normed'], kwargs['inp_extrinsics'], kwargs['inp_intrinsics'])
        rgb_embed_obs = self.view_embed_fn(rgb_obs)
        f_obs, _ = self.sample_features(x_obs_pruned, kwargs['inp_fmap'], kwargs['inp_extrinsics'], kwargs['inp_intrinsics'])
        
        u_pixel_o = torch.cat([rgb_embed_obs, f_obs], dim=-1)
        u_pixel_o = u_pixel_o * obs_view_visibility[..., None]
        
        if cfg.back_net.back_net_on or cfg.use_back_img_gt:
            back_img = kwargs['back_img'] if cfg.db not in ['humman', 'thuman1', 'zju_mocap', 'itw'] else torch.flip(kwargs['back_img'], dims=[-1])
            back_fmap = kwargs['back_fmap'] if cfg.db not in ['humman', 'thuman1', 'zju_mocap', 'itw'] else torch.flip(kwargs['back_fmap'], dims=[-1])

            rgb_back, _ = self.sample_features(x_obs_pruned, back_img, kwargs['back_extrinsics'], kwargs['back_intrinsics'])
            rgb_embed_back = self.view_embed_fn(rgb_back)
            f_back, _ = self.sample_features(x_obs_pruned, back_fmap, kwargs['back_extrinsics'], kwargs['back_intrinsics'])
            
            u_pixel_b = torch.cat([rgb_embed_back, f_back], dim=-1)
            
            u_pixel_b = u_pixel_b * (1-obs_view_visibility)[..., None]
            
        ## Set SMPL vertex based features
        if cfg.use_uv_inpainter or cfg.use_smpl_3d_feature:
            vertices_obs = deformer.vertices_obs
            
            rgb_smpl_obs, _ = self.sample_features(vertices_obs, kwargs['inp_img_normed'], kwargs['inp_extrinsics'], kwargs['inp_intrinsics'])
            rgb_smpl_embed_obs = self.view_embed_fn(rgb_smpl_obs)
            f_smpl_obs, _ = self.sample_features(vertices_obs, kwargs['inp_fmap'], kwargs['inp_extrinsics'], kwargs['inp_intrinsics'])

            smpl_o = torch.cat([rgb_smpl_embed_obs, f_smpl_obs], dim=-1)
            
            if cfg.back_net.back_net_on or cfg.use_back_img_gt:
                vertices_obs = deformer.vertices_obs
                rgb_smpl_back, _ = self.sample_features(vertices_obs, back_img, kwargs['back_extrinsics'], kwargs['back_intrinsics'])
                rgb_smpl_embed_back = self.view_embed_fn(rgb_smpl_back)
                f_smpl_back, _ = self.sample_features(vertices_obs, back_fmap, kwargs['back_extrinsics'], kwargs['back_intrinsics'])

                smpl_b = torch.cat([rgb_smpl_embed_back, f_smpl_back], dim=-1)
        
        ## Get UV map and UV features
        if cfg.use_uv_inpainter:
            uv_partial_map = self.UV_generator.get_partial_uv_map(rgb_smpl_o=rgb_smpl_obs,
                                                 rgb_smpl_b=rgb_smpl_back if cfg.back_net.back_net_on or cfg.use_back_img_gt else None,
                                                 smpl_vis_o=obs_smpl_vis_mask.long(),
                                                #  smpl_vis_b=back_smpl_vis_mask.long(),
                                                 verts_obs=deformer.vertices_obs,
                                                 data=kwargs)
            
            uv_feature_pred, uv_color_pred = self.UV_generator(uv_partial_map)
            
            closest_index = closest_verts_index_[valid_queries_mask] # Indicates where pruned sample point are assigned for SMPL vertices
            uv_feature = self.UV_generator.get_feature_from_uv(closest_index, uv_feature_pred, uv_color_pred, self.view_embed_fn)
            
            u_uv = uv_feature * (1-obs_view_visibility)[..., None] # cut out visible points

        if cfg.use_smpl_3d_feature and x_skel_pruned.numel() != 0:
            smpl_surface_features = torch.zeros((6890,self.smpl_feat_decoder.in_ch)).to(smpl_o.device)
            smpl_surface_features[obs_smpl_vis_mask.long() == 1] = smpl_o[0][obs_smpl_vis_mask.long() == 1]
            if cfg.back_net.back_net_on or cfg.use_back_img_gt:
                smpl_surface_features[obs_smpl_vis_mask.long() == 0] = smpl_b[0][obs_smpl_vis_mask.long() == 0]
            feature_3d = self.smpl_feat_decoder(smpl_surface_features, x_skel_pruned)
        
        raws_l = []
        for i in range(0, x_skel_pruned.shape[0], chunk):
            start = i
            end = i + chunk
            if end > x_skel_pruned.shape[0]:
                end = x_skel_pruned.shape[0]
            
            cnl_pts = x_skel_pruned[start:end]
            
            kwargs['u_o'] = u_pixel_o[0][start:end]
            kwargs['u_b'] = u_pixel_b[0][start:end] if cfg.back_net.back_net_on or cfg.use_back_img_gt else None
            kwargs['u_smpl_3d'] = feature_3d[0][start:end] if cfg.use_smpl_3d_feature else None
            kwargs['uv_feature'] = u_uv[0][start:end] if cfg.use_uv_inpainter else None
            kwargs['obs_view_visibility'] = obs_view_visibility[0][start:end]

            raw = self.apply_mlp(cnl_pts=cnl_pts,
                                  pos_embed_fn=pos_embed_fn,
                                  data=kwargs)
            
            raws_l.append(raw)
        
        assert valid_queries_mask.shape[0] == N_rays * N_samples

        pts_mask = valid_queries_mask
        
        raws = torch.zeros((N_rays*N_samples, 4)).to(pts)
        raws_ = torch.cat(raws_l, dim=0)
        
        raws[pts_mask == 1] = raws_
        raws[pts_mask == 0, 3] = -80
        
        raws = raws.view(N_rays, N_samples, 4)
        pts_mask = pts_mask.view(N_rays, N_samples, 1)

        rgb_map, acc_map, _, depth_map, alpha = \
            self._raw2outputs(raws, pts_mask, z_vals, rays_d, bgcolor)
        
        out = {'rgb' : rgb_map,  
                'alpha' : acc_map, 
                }
        
        if cfg.use_uv_inpainter:
            uv_map_pred = uv_color_pred
        else:
            uv_map_pred = None

        if 'x_skel' in output_list:
            out.update({'x_skel': x_skel})
        
        return out, uv_map_pred

    def apply_mlp(
            self, 
            cnl_pts, 
            pos_embed_fn,
            data=None):
        
        cnl_xyz = cnl_pts
        
        u_pixel_o = data['u_o']
        u_pixel_b = data['u_b']
        u_smpl_3d = data['u_smpl_3d']
        obs_view_visibility = data['obs_view_visibility']
        uv_feat = data['uv_feature']
        
        u_o = u_pixel_o
        
        if cfg.back_net.back_net_on or cfg.use_back_img_gt:
            u_b = u_pixel_b
            if cfg.use_uv_inpainter:
                u_b = u_b + uv_feat

        else: # No back image
            if cfg.use_uv_inpainter: # Use uv feature as invisible region feature
                u_b = uv_feat
        
        u_pix = self.lin_pix_aligned(u_o)
        
        if (cfg.back_net.back_net_on or cfg.use_back_img_gt) or cfg.use_uv_inpainter:
            u_inv = self.lin_invisible(u_b)
            u_pix[~obs_view_visibility.bool()] = u_inv[~obs_view_visibility.bool()]

        if cfg.use_smpl_3d_feature:
            u_smpl_3d = self.mlp_project_3d(u_smpl_3d)
            query_feat = u_pix + u_smpl_3d
        else:
            query_feat = u_pix
        
        f_geo = query_feat
        f_color = query_feat

        cnl_xyz_embedded = pos_embed_fn(cnl_xyz)
        raws = self.cnl_mlp(pos_embed=cnl_xyz_embedded, f_geo=f_geo, f_color=f_color)
        return raws

    def motion_infos_forward(self,
                             target_poses_69,
                             iter_val=1e7,
                             **kwargs):
        device = target_poses_69.device
        motion_info = {'iter_val': iter_val,
                       "pos_embed_fn": self.pos_embed_fn}

        target_poses = target_poses_69[None, ...]
        target_betas = kwargs['target_betas'][None]
        obs_poses = kwargs['inp_poses_69'][None]
        obs_betas = kwargs['inp_betas'][None]
        
        smpl_params = {'global_orient': torch.zeros((1,3), device=device).float(), 
                        'transl': torch.zeros((1,3), device=device).float(), 'betas': target_betas,
                        'body_pose': target_poses}
        smpl_params_obs = {'global_orient': torch.zeros((1,3), device=device).float(),
                                'transl': torch.zeros((1,3), device=device).float(), 'betas': obs_betas,
                                'body_pose': obs_poses}
        
        self.deformer.prepare_deformer(smpl_params, smpl_params_obs)
        
        if cfg.use_smpl_3d_feature:
            self.smpl_feat_decoder.prepare_decoder(self.deformer.vertices_cnl)
        return motion_info
    
    def forward(self, rays_o, rays_d, pts, z_vals, bgcolor, **kwargs):
        render_outputs, uv_map_pred = self._render_rays(rays_o, rays_d, pts, z_vals, bgcolor, **kwargs)
        rays_shape = rays_o.shape
        for k in render_outputs:
            k_shape = list(rays_shape[:-1]) + list(render_outputs[k].shape[1:])
            render_outputs[k] = torch.reshape(render_outputs[k], k_shape)
        if uv_map_pred is not None:
            render_outputs['uv_map_pred'] = uv_map_pred
        return render_outputs
    