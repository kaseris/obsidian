import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

import config
from registry import Registry

BACKBONES = Registry()
MODELS = Registry()
CLS_HEADS = Registry()


@BACKBONES.register('resnet_18')
def resnet_18():
    return models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

@BACKBONES.register('resnet_50')
def resnet_50():
    return models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

@BACKBONES.register('vit_b_16')
def vit_b_16():
    return models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)

@CLS_HEADS.register('linear')
def linear_cls_head(fan_in, embedding_sz, num_classes):
    return [nn.Linear(fan_in, embedding_sz),
            nn.ReLU(),
            nn.Linear(embedding_sz, num_classes)]
    

@MODELS.register('ResNetDeepFashion')
class ResNetDeepFashion(nn.Module):
    def __init__(self,
                 backbone: str,
                 cls_head_type: str,
                 embedding_sz: int):
        super(ResNetDeepFashion, self).__init__()
        self.resnet = BACKBONES[backbone]()
        self.cls_head_type = cls_head_type
        self.embedding_sz = embedding_sz
        self._prepare_model()

    def _prepare_model(self):
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(*CLS_HEADS[self.cls_head_type](num_features,
                                                                      self.embedding_sz,
                                                                      config.DEEP_FASHION_N_CLASSES))
        
    def forward(self):
        pass
    
if __name__ == '__main__':
    model = ResNetDeepFashion(backbone='resnet_18', cls_head_type='linear', embedding_sz=512)
    print(model)