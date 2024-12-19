import torch
import torch.nn as nn

class ResidualConv2d(nn.Module):
    def __init__(self, in_chns, dropout=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_chns)
        self.conv1 = nn.Conv2d(in_chns, in_chns, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_chns, in_chns, kernel_size=3, stride=1, padding=1, bias=True)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3) if dropout else nn.Identity()

        self.norm2 = nn.GroupNorm(8, in_chns)
        self.conv3 = nn.Conv2d(in_chns, in_chns, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv4 = nn.Conv2d(in_chns, in_chns, kernel_size=3, stride=1, padding=1, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, in_chns, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_chns, 1, 1)), requires_grad=True)

    def forward(self, inp):
        
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)

        x = self.dropout(x)

        y = inp + x * self.beta  # Residual

        x = self.norm2(y)
        x = self.conv3(x)
        x = self.gelu(x)
        x = self.conv4(x)

        x = self.gelu(x)
        x = self.dropout(x)

        x = self.gamma * x + y  # Residual

        return x


class DownConv2d(nn.Module):
    def __init__(self, in_chns, out_chns, kernel_size=3, stride=2,
                 norm=nn.Identity(), activation=False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.downconv = nn.Conv2d(in_chns, out_chns, kernel_size=kernel_size,
                                  stride=stride, padding=padding, bias=True)
        self.norm = norm
        self.activation = nn.GELU() if activation else nn.Identity()

    def forward(self, x):
        x = self.downconv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class UpConv2d(nn.Module):
    def __init__(self, in_chns, out_chns, out_pad=1,
                 norm=nn.Identity(), activation=False):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_chns, out_chns, kernel_size=3, stride=2, padding=1,
                                         output_padding=out_pad, bias=True)
        self.norm = norm
        self.activation = nn.GELU() if activation else nn.Identity()

    def forward(self, x):
        x = self.upconv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

# ================== Source Code from https://github.com/megvii-research/NAFNet ================== #

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

# ================== Source Code from https://github.com/megvii-research/NAFNet ================== #