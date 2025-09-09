import torch
from torch import nn
from core.nets.human_nerf.sparse_conv_net import SparseConvNet
import spconv.pytorch as spconv

class SMPL_Feature_Volume(nn.Module):
    def __init__(self, smpl_model, in_ch):
        super(SMPL_Feature_Volume, self).__init__()
        
        self.smpl_model = smpl_model
        self.smpl_faces = torch.from_numpy(self.smpl_model.faces.astype(int)).to('cuda')
        # load spconv
        self.mlp = nn.Sequential(nn.Linear(in_ch, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))
        self.encoder_3d = SparseConvNet(in_ch=32, num_layers=4)
        self.in_ch = in_ch
        
    def prepare_decoder(self, cnl_verts):
        self.tpose_sparse_volume_dict, _ = self.make_canonical_space_sparse_conv_tensor_volume(cnl_verts) # out shape : [96,384,384]
    
    def create_sparse_tensor(self, feat):
        # feat : (N, C)
        # coord : (N, 4) batch_idx, z, y, x
        # out_sh : (3)
        # batch_size : (1)
        if feat.ndim == 3:
            feat = feat.view(-1, feat.shape[-1])
        
        coord = self.tpose_sparse_volume_dict['coord'].clone()
        out_sh = self.tpose_sparse_volume_dict['out_sh']
        batch_size = self.tpose_sparse_volume_dict['batch_size']
        
        return spconv.core.SparseConvTensor(feat, coord, out_sh, batch_size)
    
    def forward(self, features, canonical_pts):
        # features : (N, C)
        x = self.mlp(features)
        inp_features = self.create_sparse_tensor(x)
        canonical_pts_grid = self.get_voxel_coords_of_query(canonical_pts)
        canonical_pts_grid = canonical_pts_grid[:,None,None]
        out_features = self.encoder_3d(inp_features, canonical_pts_grid)
        
        return out_features

    def make_canonical_space_sparse_conv_tensor_volume(self, tpose_smpl_verts):
        self.big_box = True
        # obtain the bounds for coord construction
        min_xyz = torch.min(tpose_smpl_verts, dim=1)[0]
        max_xyz = torch.max(tpose_smpl_verts, dim=1)[0]

        if self.big_box:  # False
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[:, 2] -= 0.05
            max_xyz[:, 2] += 0.05
        # size of canonical smpl bbox
        bounds = torch.cat([min_xyz.unsqueeze(1), max_xyz.unsqueeze(1)], axis=1) # torch.Size([1, 2, 3])
        
        dhw = tpose_smpl_verts[:, :, [2, 1, 0]]
        min_dhw = min_xyz[:, [2, 1, 0]]
        max_dhw = max_xyz[:, [2, 1, 0]] # permuting
        voxel_size = torch.Tensor([0.005, 0.005, 0.005]).to(device=dhw.device)
        coord = torch.round((dhw - min_dhw.unsqueeze(1)) / voxel_size).to(dtype=torch.int32) # voxel coordinate of the smpl verts
        
        # construct the output shape of volume space
        out_sh = torch.ceil((max_dhw - min_dhw) / voxel_size).to(dtype=torch.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        sp_input = {}

        # coordinate: [N, 4], batch_idx, z, y, x 
        sh = dhw.shape # torch.Size([1, 6890, 3])
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(coord)
        coord = coord.view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(out_sh, dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]
        sp_input['bounds'] = bounds
        
        return sp_input, _#, pc_features
    
    def get_voxel_coords_of_query(self, cnl_pts):
        # convert xyz to the voxel coordinate dhw
        pts = cnl_pts
        dhw = pts[..., [2, 1, 0]]
        min_dhw = self.tpose_sparse_volume_dict['bounds'][:, 0, [2, 1, 0]] # pick min coord and permute
        dhw = dhw - min_dhw[:, None]
        dhw = dhw / torch.tensor([0.005, 0.005, 0.005]).to(dhw)
        
        out_sh = torch.tensor(self.tpose_sparse_volume_dict['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1 # normalize to [-1, 1]
        
        # permute back normalized coordinates into xyz style
        # these coordinates will be used for grid_sample() method
        grid_coords = dhw[..., [2, 1, 0]]
        
        return grid_coords
    
    def compute_normal(self, vertices, faces):
        def normalize_v3(arr):
            ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
            lens = torch.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2 + arr[..., 2] ** 2)
            eps = 0.00000001
            lens[lens < eps] = eps 
            arr[..., 0] /= lens
            arr[..., 1] /= lens
            arr[..., 2] /= lens
            return arr
        
        norm = torch.zeros(vertices.shape, dtype=vertices.dtype).cuda()
        tris = vertices[:, faces] # [bs, 13776, 3, 3]
        n = torch.cross(tris[:, :, 1] - tris[:, :, 0], tris[:, :, 2] - tris[:, :, 0]) 
        n = normalize_v3(n)
        norm[:, faces[:, 0]] += n
        norm[:, faces[:, 1]] += n
        norm[:, faces[:, 2]] += n
        norm = normalize_v3(norm)

        return norm