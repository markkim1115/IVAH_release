import torch.nn as nn
from core.nets.human_nerf.resnet.resnet import Backbone

class ImageEncoder(nn.Module):
    def __init__(self, model_name='resnet34'):
        super(ImageEncoder, self).__init__()
        self.backbone = Backbone(model_name=model_name)
    
    def forward(self, image):
        # image (3,H,W)
        out_data = {}
        if image.ndim == 3:
            image = image.unsqueeze(0) # (N,3,H,W)
        
        fmap = self.backbone.forward(image)
        out_data['fmap'] = fmap
        
        return out_data
