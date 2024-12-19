import os
import torch
import torch.nn as nn
from core.nets.human_nerf.back_net.components import DownConv2d, ResidualConv2d, UpConv2d
from core.utils.body_util import load_obj
from core.utils.network_util import init_weights
from configs.config import cfg

class UNet(nn.Module):
    def __init__(self, in_chns=3):
        super().__init__()
        self.downconv1 = DownConv2d(in_chns, 64)
        self.residual_conv1 = ResidualConv2d(64, dropout=True)

        self.downconv2 = DownConv2d(64, 128)
        self.residual_conv2 = ResidualConv2d(128, dropout=False)

        self.downconv3 = DownConv2d(128, 256)
        self.residual_conv3 = ResidualConv2d(256, dropout=False)

        self.downconv4 = DownConv2d(256, 512)
        self.residual_conv4 = ResidualConv2d(512, dropout=False)

        self.downconv5 = DownConv2d(512, 512)
        self.residual_conv5 = ResidualConv2d(512, dropout=False)

        self.residual_conv6 = ResidualConv2d(512, dropout=False)
        self.residual_conv7 = ResidualConv2d(512, dropout=False)
        self.residual_conv8 = ResidualConv2d(512, dropout=False)

        self.upconv1 = UpConv2d(512 + 512, 512, activation=False)
        self.residual_conv9 = ResidualConv2d(512, dropout=False)
        self.upconv2 = UpConv2d(512 + 512, 512, activation=False)
        self.residual_conv10 = ResidualConv2d(512, dropout=False)
        self.upconv3 = UpConv2d(512 + 256, 256, activation=False)
        self.residual_conv11 = ResidualConv2d(256, dropout=False)
        self.upconv4 = UpConv2d(256 + 128, 128, activation=False)
        self.residual_conv12 = ResidualConv2d(128, dropout=False)
        self.upconv5 = UpConv2d(128 + 64, 64, activation=False)
        
        
        self.rgb_layer = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                                                nn.GELU(),
                                                nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                                nn.Sigmoid())
        self.feature_extract = nn.Sequential(
                                        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                                        nn.GELU(),
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
                                        )
        self.gelu = nn.GELU()
        
    def forward(self, x, layers=[], encode_only=False):
        
        x1 = self.downconv1(x)      # 256
        x = self.residual_conv1(x1)
        x2 = self.downconv2(x)      # 128
        x = self.residual_conv2(x2)
        x3 = self.downconv3(x)      # 64
        x = self.residual_conv3(x3)
        x4 = self.downconv4(x)      # 32
        x = self.residual_conv4(x4)
        x5 = self.downconv5(x)      # 16
        x = self.residual_conv5(x5)
        
        x = self.residual_conv6(x)
        x = self.gelu(x)
        x = self.residual_conv7(x)
        x = self.gelu(x)
        x = self.residual_conv8(x)
        x = self.gelu(x)

        x = torch.cat([x, x5], dim=1)
        x = self.upconv1(x)         # 32
        x = self.residual_conv9(x)
        x = torch.cat([x, x4], dim=1)
        x = self.upconv2(x)
        x = self.residual_conv10(x)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv3(x)
        x = self.residual_conv11(x)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv4(x)
        x = self.residual_conv12(x)
        x = torch.cat([x, x1], dim=1)
        x = self.upconv5(x)
        
        texture = self.rgb_layer(x)
        feature_uv = self.feature_extract(texture)
        
        return feature_uv, texture
    
class UV_Generator(nn.Module):
    def __init__(self, cfg, in_channels=3, uv_map_size=256):
        super().__init__()
        self.cfg = cfg

        # UV map data fetching
        self.uv_map_size = uv_map_size
        uv_obj_file_path = 'third_parties/smpl/models/smpl_uv.obj'
        smpl_uv_data = load_obj(uv_obj_file_path)
        self.load_uv(smpl_uv_data)
        bc_pickle = 'third_parties/smpl/models'+f'/barycentric_h{uv_map_size:04d}_w{uv_map_size:04d}.pkl'
        if not os.path.exists(bc_pickle):
            print(f"barycentric_h{uv_map_size:04d}_w{uv_map_size:04d}.pkl does not exist. Please generate the file.")
            exit(0)
        else:
            import pickle
            with open(bc_pickle, 'rb') as f:
                bary_dict = pickle.load(f)
                bary_id = bary_dict['face_id']
                bary_weights = bary_dict['bary_weights']
                edge_dict = bary_dict['edge_dict']
            
            bary_id = torch.from_numpy(bary_id).long()
            bary_weights = torch.from_numpy(bary_weights).float()

            self.register_buffer('bary_id', bary_id)
            self.register_buffer('bary_weights', bary_weights)
            
        # U-net
        self.inpainter = UNet(in_chns=in_channels)
        
        init_weights(self)
        
    def forward(self, x):
        uv_feature, uv_rgb = self.inpainter(x, layers=[], encode_only=False)

        return uv_feature, uv_rgb

    def perspective_projection(self, pts, extr, intr):
        # Args
        # extr : (N, 4, 4)
        # intr : (N, 3, 3)
        # pts : (N, P, 3)
        # Return
        # pts_2d : (N, P, 2)
        R = extr[:, :3, :3]
        T = extr[:, :3, 3]
        pts_cam = pts @ R.permute(0,2,1) + T
        pts_uvz = pts_cam @ intr.permute(0,2,1)
        pts_2d = pts_uvz[:, :, :2] / pts_uvz[:, :, 2:]
        
        return pts_2d
    
    def barycentric_interpolation(self, vert_colors):
        vert_colors = vert_colors[...,:][self.vt_to_v_index]
        triangle_values = vert_colors[self.faces_uv][self.bary_id,:,:]
        bw = self.bary_weights[:,:,None,:]
        im = torch.matmul(bw, triangle_values).squeeze(2)
        im = torch.clip(im, 0, 1)
        return im
    
    def get_partial_uv_map(self,rgb_smpl_o, smpl_vis_o, rgb_smpl_b=None, smpl_vis_b=None, verts_obs=None, data=None):
        verts_feat = torch.zeros((6890, 3), dtype=torch.float32, device=rgb_smpl_o.device)
        verts_feat[smpl_vis_o == 1] = rgb_smpl_o[0][smpl_vis_o == 1]

        if cfg.back_net.back_net_on:
            verts_feat[smpl_vis_o == 0] = rgb_smpl_b[0][smpl_vis_o == 0]
            
        uv_map_partial = torch.zeros((self.uv_map_size, self.uv_map_size, 3), dtype=torch.float32, device=rgb_smpl_o.device)
        uv_map_partial = self.barycentric_interpolation(verts_feat)
        
        debug = False
        if debug:
            from PIL import Image
            import numpy as np
            from core.utils.image_util import to_8b_image
            # from core.utils.vis_util import draw_2D_joints
            # vertices = verts_obs
            vis = uv_map_partial.detach().cpu().numpy()[:,:,:3]
            vis = (vis * 255.).astype(np.uint8)
            vis = Image.fromarray(vis).resize((512,512))
            vis = np.array(vis)
            tempobsimg = data['inp_img_normed'][0].permute(1,2,0).detach().cpu().numpy()
            tempobsimg = (tempobsimg * 255.).astype(np.uint8)
            tempbackimg = data['back_img'][0].permute(1,2,0).detach().cpu().numpy()
            tempbackimg = (tempbackimg * 255.).astype(np.uint8)
            gt = to_8b_image(data['uv_map_gt'].data.detach().cpu().numpy())
            gt = Image.fromarray(gt).resize((512,512))
            gt = np.array(gt)
            # obs_E = data['inp_extrinsics'][None]
            # obs_intr = data['inp_intrinsics'][None]
            
            # obs_verts_2d_ = self.perspective_projection(vertices[:,smpl_vis_o == 1], obs_E, obs_intr)[0]
            # tempobsimg_ = draw_2D_joints(tempobsimg, obs_verts_2d_.detach().cpu().numpy())
            # vis = tempobsimg_
            vis = np.concatenate([vis, gt], axis=1)
            Image.fromarray(vis).save('interp_partial_uv_debug.png')
            
        
        uv_map_partial = uv_map_partial.permute(2,0,1)[None] # (1, C, H, W)
        return uv_map_partial
    
    def load_uv(self, data):
        uv = data['uv']
        uv = uv * (self.uv_map_size - 1)
        uv = uv.astype(int)
        uv[:, 1] = (self.uv_map_size - 1) - uv[:, 1]
        uv_mapping = torch.from_numpy(uv)
        faces_uv = torch.from_numpy(data['faces_uv'])
        vt_to_v = data['vt_to_v']
        vt_to_v_index = torch.tensor([vt_to_v[idx] for idx in range(7576)], requires_grad=False).long()

        self.register_buffer('uv_mapping', uv_mapping)
        self.register_buffer('faces_uv', faces_uv)
        self.register_buffer('vt_to_v_index', vt_to_v_index)
    
    def get_feature_from_uv(self,
                            closest_index, # (n_pts)
                            inpainted_uv_feat, # (1, 32, H, W)
                            inpainted_uv_rgb, # (1, 3, H, W)
                            color_embedder,
                            ):
        # closest_index: (N,)
        # inpainted_uv_feat: (1, 32, H, W)
        # inpainted_uv_rgb: (1, 3, H, W)
        # color_embedder: nn.Module
        
        uv_coordinates = self.uv_mapping[closest_index] # (N, 2)
        resampled_uv_rgb = inpainted_uv_rgb[:,:,uv_coordinates[:,1],uv_coordinates[:,0]] # (1, 3, N)
        resampled_uv_feat = inpainted_uv_feat[:,:,uv_coordinates[:,1],uv_coordinates[:,0]] # (1, 32, N)
        
        resampled_uv_rgb = resampled_uv_rgb.permute(0,2,1)[0] # (N, 3)
        resampled_uv_feat = resampled_uv_feat.permute(0,2,1)[0] # (N, 32)
        
        # embed the rgb
        embedded_color = color_embedder(resampled_uv_rgb)
        sample_wise_feat = torch.cat([embedded_color, resampled_uv_feat], dim=1)
        
        return sample_wise_feat