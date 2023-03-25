import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

from registry import Registry

BACKBONES = Registry()
MODELS = Registry()

@BACKBONES.register('resnet_18')
def resnet_18():
    return models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

@BACKBONES.register('resnet_50')
def resnet_50():
    return models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

@BACKBONES.register('vit_b_16')
def vit_b_16():
    return models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)

@MODELS.register('ResNetDeepFashion')
class ResNetDeepFashion(nn.Module):
    def __init__(self,
                 backbone: str):
        super(ResNetDeepFashion, self).__init__()
        self.resnet = BACKBONES[backbone]()
        
    def forward(self):
        pass
    