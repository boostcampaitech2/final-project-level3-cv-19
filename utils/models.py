import torch
import yaml, os
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
import segmentation_models_pytorch as smp

class ResNet50(nn.Module):
    def __init__(self,pretrained=True):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(in_features=2048, out_features=5, bias=True)
    
    def forward(self, x):
        return self.model(x)

class ResNet18(nn.Module):
    def __init__(self,pretrained=True):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(in_features=2048, out_features=5, bias=True)
    
    def forward(self, x):
        return self.model(x)


class UnetResnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnet50", 
                        encoder_weights="imagenet",     
                        in_channels=3,                  
                        classes=2,)                    

    def forward(self, x):
        return self.model(x)

class UnetResnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnet18", 
                        encoder_weights="imagenet",     
                        in_channels=3,                  
                        classes=6,)                    

    def forward(self, x):
        return self.model(x)



# ================================== fcn_resnet50 ====================================
class FcnResnet50(nn.Module):
    '''
    fcn_resnet50
    '''
    def __init__(self, num_classes=11):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']

# ============================== efficientnet_b0_Unet ================================
class EfficientnetB0(nn.Module):
    def __init__(self):
        super().__init__()
        model = smp.Unet(encoder_name="efficientnet-b0", 
                        encoder_weights="imagenet",     
                        in_channels=3,                  
                        classes=11,)                    

    def forward(self, x):
        return self.model(x)



# ================================== hrnet_unet ======================================
class HrnetW48(nn.Module):
    '''
    AI stage 토론 게시판
    https://stages.ai/competitions/78/discussion/talk/post/809
    '''
    def __init__(self, num_classes=11):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="tu-hrnet_w48",       
            encoder_weights="imagenet",   
            in_channels=3,                
            classes=11,)

    def forward(self, x):
        return self.model(x)


# ================================== Unet++ efficientnet-b8 ======================================
class UnetPlusPlusB8(nn.Module):
    '''
    AI stage 토론 게시판
    https://stages.ai/competitions/78/discussion/talk/post/809
    '''
    def __init__(self, num_classes=11):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b8",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            )
    def forward(self, x):
        return self.model(x)


# ================================== Unet++ efficientnet-b7 ======================================
class UnetPlusPlusB7(nn.Module):
    '''
    AI stage 토론 게시판
    https://stages.ai/competitions/78/discussion/talk/post/809
    '''
    def __init__(self, num_classes=11):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7",
            encoder_weights="imagenet",
            in_channels=3,  
            classes=1,  
        )
    def forward(self, x):
        return self.model(x)

