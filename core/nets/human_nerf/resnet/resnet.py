# Copyright (c) 2019, University of Pennsylvania, Max Planck Institute for Intelligent Systems
# This script is borrowed and extended from SPIN
import torch.nn as nn
import torchvision.models.resnet as resnet
from core.utils.misc import count_parameters

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1,downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
    
        if self.downsample is not None:
            identity = self.downsample(x)
    
        out += identity
        
        out = self.relu(out)
        

        return out

class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, layers, block=BasicBlock):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(fmap)
        x_layer1 = self.layer1(x) # 64
        # x = self.layer2(x) # 128
        # x = self.layer3(x) # 256
        # x = self.layer4(x) # 512

        # xavg = self.avgpool(x)
        # global_feat = xavg.view(xavg.size(0), -1)

        # fmap_size = latents[0].shape[-2:]
        # for idx in range(len(latents)):
        #     latents[idx] = F.interpolate(latents[idx], size=(fmap_size), mode='bilinear', align_corners=True)
        # fmap = torch.cat(latents, dim=1)
        fmap = x_layer1
        
        # return fmap, global_feat
        return fmap

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

class Backbone(nn.Module):
    def __init__(self, model_name='resnet34'):
        super(Backbone, self).__init__()
        self.model_name = model_name
        self.backbone = self.load_model()
    
    def forward(self, x):
        fmap = self.backbone.forward(x)
        return fmap
    
    def load_model(self):
        if self.model_name == 'resnet34':
            net = load_resnet34()
        elif self.model_name == 'resnet18':
            net = load_resnet18()
        else:
            print(f"Image Encoder backbone model name {self.model_name} not supported")
            exit(0)
        print("Number of parameters of backbone(image encoder): {} (total, trainable)".format(count_parameters(net)))
        return net

def load_resnet18():
    model = ResNet([2,2,2,2])
    state = resnet.resnet18(pretrained=True).state_dict()
    model.load_state_dict(state, strict=False)
    return model

def load_resnet34():
    model = ResNet([3,4,6,3])
    state = resnet.resnet34(pretrained=True).state_dict()
    model.load_state_dict(state, strict=False)
    return model

# import torch
# net = load_resnet18()
# inp = torch.randn(1,3,512,512)
# with torch.no_grad():
#     fmap, fvec = net(inp)
# pdb.set_trace()
