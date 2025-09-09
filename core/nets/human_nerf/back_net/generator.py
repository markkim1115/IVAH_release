import torch
import torch.nn as nn
from core.nets.human_nerf.back_net.components import DownConv2d, ResidualConv2d, UpConv2d

class Generator(nn.Module):
    def __init__(self, in_chns=3, out_chns=3):
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

        self.last_conv = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(32, out_chns, kernel_size=1, stride=1, padding=0),
                                       nn.Sigmoid())

        self.gelu = nn.GELU()
    
    def forward(self, x, **_):
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

        x = self.last_conv(x)
        return x
    
def load_pretrained(net:nn.Module, ckpt_path:str):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['ckpt']
    state_dict = {k[10:]:v for k,v in state_dict.items()} # Essential! Removing 'generator.' from parameter keys make the ckpt work! :(
    net.load_state_dict(state_dict, strict=False)
    print('Pre-trained back image generator loaded.')