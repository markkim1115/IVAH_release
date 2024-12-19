import torch
from torch import nn
import torch.nn.functional as F
import spconv.pytorch as spconv

def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    # tmp = spconv.SubMConv3d(in_channels,
    #                       out_channels,
    #                       3,
    #                       bias=False,
    #                       indice_key=indice_key)
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())

class SparseConvNet(nn.Module):
    """Find the corresponding 3D feature of query point along the ray
    
    Attributes:
        conv: sparse convolutional layer 
        down: sparse convolutional layer with downsample 
    """
    def __init__(self, in_ch, num_layers=4):
        super(SparseConvNet, self).__init__()
        self.in_ch = in_ch
        
        self.num_layers = num_layers

        self.conv0 = double_conv(in_ch, 32, 'subm0')
        self.down0 = stride_conv(32, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 96, 'down2')

        self.conv3 = triple_conv(96, 96, 'subm3')
        self.down3 = stride_conv(96, 96, 'down3')

        self.conv4 = triple_conv(96, 96, 'subm4')

    def forward(self, x, point_normalied_coords):
        """Find the corresponding 3D feature of query point along the ray.

        Args:
            x: Sparse Conv Tensor
            point_normalied_coords: Voxel grid coordinate, integer normalied to [-1, 1]
        
        Returns:
            features: Corresponding 3D feature of query point along the ray
        """
        features = []

        net = self.conv0(x)
        net = self.down0(net)
        
        # point_normalied_coords = point_normalied_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        if self.num_layers > 1:
            net = self.conv1(net)
            net1 = net.dense()
            # torch.Size([1, 32, 1, 1, 4096])
            feature_1 = F.grid_sample(net1, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_1)
            self.channel = 32
            net = self.down1(net)
        
        if self.num_layers > 2:
            net = self.conv2(net)
            net2 = net.dense()
            # torch.Size([1, 64, 1, 1, 4096])
            feature_2 = F.grid_sample(net2, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_2)
            self.channel = 64
            net = self.down2(net)
        
        if self.num_layers > 3:
            net = self.conv3(net)
            net3 = net.dense()
            # 128
            feature_3 = F.grid_sample(net3, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_3)
            self.channel = 96
            net = self.down3(net)
        
        if self.num_layers > 4:
            net = self.conv4(net)
            net4 = net.dense()
            # 256
            feature_4 = F.grid_sample(net4, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_4)
        
        features = torch.cat(features, dim=1)
        
        features = features.view(features.size(0), -1, features.size(4)).transpose(1,2)

        return features
    
class SparseConvNet_Triplane(nn.Module):
    """Find the corresponding 3D feature of query point along the ray
    
    Attributes:
        conv: sparse convolutional layer 
        down: sparse convolutional layer with downsample 
    """
    def __init__(self, in_ch, out_ch=128, num_layers=4):
        super(SparseConvNet_Triplane, self).__init__()
        self.num_layers = num_layers
    
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv0 = double_conv(in_ch, 32, 'subm0')
        self.down0 = stride_conv(32, 32, 'down0')
        self.conv1 = double_conv(32, 64, 'subm1')
        self.down1 = stride_conv(64, 128, 'down1')
        self.conv2 = triple_conv(128, out_ch, 'subm2')
        
    def sample_triplane(self, coords, planes):
        # coords : (N,3)
        # planes : (1,3,32,H,W)
        # Output : f_xy, f_yz, f_zx (N,32) (N,32) (N,32)
        coords_xy = coords[...,:2].unsqueeze(0)
        coords_yz = coords[...,1:3].unsqueeze(0)
        coords_xz = coords[...,[0,2]].unsqueeze(0)
        
        f_xy = F.grid_sample(planes[0], coords_xy, mode='bilinear', padding_mode='zeros', align_corners=True)
        f_xy = f_xy.squeeze(2).permute(0,2,1).unsqueeze(1)
        f_yz = F.grid_sample(planes[1], coords_yz, mode='bilinear', padding_mode='zeros', align_corners=True)
        f_yz = f_yz.squeeze(2).permute(0,2,1).unsqueeze(1)
        f_xz = F.grid_sample(planes[2], coords_xz, mode='bilinear', padding_mode='zeros', align_corners=True)
        f_xz = f_xz.squeeze(2).permute(0,2,1).unsqueeze(1) # (N,32)
        
        f_tri = torch.cat([f_xy, f_yz, f_xz], dim=1).sum(1)

        return f_tri

    def forward(self, x, point_normalied_coords):
        """Find the corresponding 3D feature of query point along the ray.

        Args:
            x: Sparse Conv Tensor
            point_normalied_coords: Voxel grid coordinate, integer normalied to [-1, 1]
        
        Returns:
            features: Corresponding 3D feature of query point along the ray
        """
        
        x = self.conv0(x)
        x = self.down0(x)
        x = self.conv1(x)
        x = self.down1(x)
        x = self.conv2(x)

        x = x.dense()
        
        plane_xy = torch.mean(x, dim=-3, keepdim=False)
        plane_yz = torch.mean(x, dim=-1, keepdim=False)
        plane_xz = torch.mean(x, dim=-2, keepdim=False)
        
        f_tri = self.sample_triplane(point_normalied_coords, [plane_xy, plane_yz, plane_xz])
        
        # features = torch.cat(features, dim=1)
        # features = features.view(features.size(0), -1, features.size(4)).transpose(1,2)

        return f_tri