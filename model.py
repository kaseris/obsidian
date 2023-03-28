from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

import config
from registry import Registry

BACKBONES = Registry()
MODELS = Registry()
CLS_HEADS = Registry()


@BACKBONES.register('resnet_18')
def resnet_18():
    """
    Returns a ResNet18 instance pretrained on ImageNet.
    """
    return models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

@BACKBONES.register('resnet_50')
def resnet_50():
    """
    Returns a ResNet50 instance pretrained on ImageNet.
    """
    return models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

@BACKBONES.register('vit_b_16')
def vit_b_16():
    """
    Returns a vision transformer (ViT) instance pretrained on ImageNet.
    """
    return models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)

@CLS_HEADS.register('linear')
class LinearClassificationHead(nn.Module):
    """
    A classification head that consists of a linear embedding layer followed by
    a ReLU activation function and a linear classification layer.

    Args:
        fan_in (int): The number of input features for the embedding layer.
        embedding_sz (int): The number of output features for the embedding layer.
        n_classes (int): The number of output classes for the classification layer.

    Returns:
        The output tensor of the classification head and the embedding tensor.
    """
    def __init__(self, fan_in : int,
                 embedding_sz : int,
                 n_classes) -> None:
        super(LinearClassificationHead, self).__init__()
        self.embedding = nn.Linear(in_features=fan_in,
                                   out_features=embedding_sz)
        self.act = nn.ReLU()
        self.cls_head = nn.Linear(in_features=embedding_sz,
                             out_features=n_classes)
        
    def forward(self, x):
        embedding = self.embedding(x.squeeze())
        embedding = self.act(embedding)
        out_cls = self.cls_head(embedding)
        return out_cls, embedding

@MODELS.register('ResNetDeepFashion')
class ResNetDeepFashion(nn.Module):
    """
    A class that defines the ResNetDeepFashion model.

    The ResNetDeepFashion model is a modified version of the ResNet architecture
    that is optimized for the DeepFashion dataset. The model consists of a ResNet
    backbone and a classification head that is used to predict the class labels of
    the input images.

    Attributes:
    ----------
    backbone : str
        The backbone architecture to use, e.g. "resnet50".
    cls_head_type : str
        The type of classification head to use, e.g. "linear".
    attr_cls_head_type : str, None
        The type of attribute classification head to use, e.g. "linear". If `None`, the model will not predict garment attributes.
    embedding_sz : int
        The size of the output embeddings.

    Methods:
    --------
    forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Forward pass of the ResNetDeepFashion model.

    freeze_weights():
        Freezes the weights of the layers up to `self.resnet.fc`.
    """
    def __init__(self,
                 backbone: str,
                 cls_head_type : str,
                 attr_cls_head_type : Union[str, None],
                 embedding_sz: int,):
        """
        Initialize the ResNetDeepFashion model.

        Args:
        ----------
        backbone : str
            The backbone architecture to use, e.g. "resnet50".
        cls_head_type : str
            The type of classification head to use, e.g. "linear".
        embedding_sz : int
            The size of the output embeddings.
        """
        super(ResNetDeepFashion, self).__init__()
        self.cls_head_type = cls_head_type
        self.backbone = BACKBONES[backbone]()
        self.cls_head = None
        self.embedding_sz = embedding_sz
        # if attr_cls_head_type is not None:
        #     self.attr_cls_head_type = CLS_HEADS[attr_cls_head_type]()
        self._prepare_model()

    def _prepare_model(self):
        """
        Prepare the ResNetDeepFashion model by modifying the final layer.
        """
        num_features = list(self.backbone.children())[-1].in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.cls_head = CLS_HEADS[self.cls_head_type](num_features, self.embedding_sz,
                                                      config.DEEP_FASHION_N_CLASSES)

    
    def freeze_weights(self):
        """
        Freezes the weights of the layers up to `self.resnet.fc`.
        """
        for name, param in self.resnet.named_parameters():
            if 'fc' in name:
                break
            param.requires_grad = False
            
    def unfreeze_weights(self):
        """
        Freezes the weights of the layers up to `self.resnet.fc`.
        """
        for name, param in self.resnet.named_parameters():
            if 'fc' in name:
                break
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the ResNetDeepFashion model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_channels, height, width).
            targets (torch.Tensor, optional): The target tensor of shape (batch_size) containing the ground-truth
                class indices. If not None, the function will return the output tensor of the ResNet backbone, the 
                output tensor of the classification head and the classification loss.

        Returns:
            A tuple containing the output tensor of the ResNet backbone, the output tensor of the classification head
            and the classification loss if targets is not None, otherwise only the output tensor of the ResNet backbone
            and the output tensor of the classification head.
        """
        out = self.backbone(x)
        preds, embeddings = self.cls_head(out)
        preds = F.softmax(preds, dim=1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(preds, target=targets)
            return out, preds, loss
        return out, preds, loss
    