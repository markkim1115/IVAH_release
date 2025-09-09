import torch
import torch.nn as nn
from core.utils.network_util import initmod

class CanonicalMLP(nn.Module):
    def __init__(self, mlp_depth=8, mlp_width=256, 
                 input_ch=3, comps_dim=None,
                 **_):
        super(CanonicalMLP, self).__init__()

        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.input_ch = input_ch
        
        self.act = nn.ReLU()
        
        self.mlp0 = nn.Linear(input_ch, mlp_width)
        self.mlp1 = nn.Linear(mlp_width, mlp_width)
        self.mlp2 = nn.Linear(mlp_width, mlp_width)
        self.mlp3 = nn.Linear(mlp_width, mlp_width)
        self.mlp4 = nn.Linear(mlp_width, mlp_width)
        skip_layer_input_dim = input_ch + mlp_width
        self.mlp5 = nn.Linear(skip_layer_input_dim, mlp_width)
        self.mlp6 = nn.Linear(mlp_width, mlp_width)
        self.mlp7 = nn.Linear(mlp_width, mlp_width)

        self.mlp_density = nn.Linear(mlp_width, 1)
        self.mlp_feature = nn.Linear(mlp_width, mlp_width)
        
        f_pixel_dim = comps_dim['f_pixel_dim']
        
        f_color_dim = mlp_width + f_pixel_dim
        self.mlp8 = nn.Linear(f_color_dim, mlp_width//2)
        self.mlp_color = nn.Linear(mlp_width//2, 3)

        set_of_mlps = [self.mlp0, self.mlp1, self.mlp2, self.mlp3, 
                       self.mlp4, self.mlp5, self.mlp6, self.mlp7, self.mlp8, 
                       self.mlp_feature, self.mlp_color, self.mlp_density
                       ]
        
        self.init_mlps(set_of_mlps)
        
    def init_mlps(self, set_of_mlps):
        gain_calc = None
        
        if isinstance(self.act, nn.ReLU):
            gain_calc = nn.init.calculate_gain('relu')
        elif isinstance(self.act, nn.LeakyReLU):
            gain_calc = nn.init.calculate_gain('leaky_relu')
        elif isinstance(self.act, nn.Softplus):
            gain_calc = nn.init.calculate_gain('softplus')
        
        for i in range(len(set_of_mlps)-1):
            initmod(set_of_mlps[i], gain_calc)
        
        initmod(self.mlp_density)
        initmod(self.mlp_color)

    def forward(self, pos_embed, f_pixel=None, f_geo=None, f_color=None, f_3d=None, **_):
        feat_geo = f_geo if (f_pixel is None) and (not f_geo is None) else f_pixel
        feat_color = f_color if (f_pixel is None) and (not f_color is None) else f_pixel
        if pos_embed is None:
            inp = feat_geo
        else:
            inp = torch.cat([pos_embed, feat_geo], dim=-1)
        if not f_3d is None:
            inp = torch.cat([inp, f_3d], dim=-1)
        x = self.mlp0(inp)
        x = self.act(x)
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        x = self.act(x)
        x = self.mlp3(x)
        x = self.act(x)
        x = self.mlp4(x)
        x = self.act(x)
        x = torch.cat([inp, x], dim=-1)
        
        x = self.mlp5(x)
        x = self.act(x)
        x = self.mlp6(x)
        x = self.act(x)
        x = self.mlp7(x)
        x = self.act(x)
        sigma = self.mlp_density(x)
        h = self.mlp_feature(x)
        
        h = torch.cat([h, feat_color], dim=-1)
        if not f_3d is None:
            h = torch.cat([h, f_3d], dim=-1)

        h = self.mlp8(h)
        h = self.act(h)
        rgb = self.mlp_color(h)

        outputs = torch.cat([rgb, sigma], dim=-1)
        
        return outputs
        