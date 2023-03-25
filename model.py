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
def linear_cls_head(fan_in, embedding_sz, num_classes):
    """
    Returns a linear classification head composed of two fully connected layers with ReLU activation in between.

    Args:
        fan_in (int): The number of input features.
        embedding_sz (int): The size of the embedding layer.
        num_classes (int): The number of output classes.

    Returns:
        List[nn.Module]: A list of the fully connected layers and ReLU activation layer.
    """
    return [nn.Linear(fan_in, embedding_sz),
            nn.ReLU(),
            nn.Linear(embedding_sz, num_classes)]
    

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
                 cls_head_type: str,
                 embedding_sz: int):
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
        self.resnet = BACKBONES[backbone]()
        self.cls_head_type = cls_head_type
        self.embedding_sz = embedding_sz
        self._prepare_model()

    def _prepare_model(self):
        """
        Prepare the ResNetDeepFashion model by modifying the final layer.
        """
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(*CLS_HEADS[self.cls_head_type](num_features,
                                                                      self.embedding_sz,
                                                                      config.DEEP_FASHION_N_CLASSES))
    
    def freeze_weights(self):
        """
        Freezes the weights of the layers up to `self.resnet.fc`.
        """
        for name, param in self.resnet.named_parameters():
            if 'fc' in name:
                break
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the ResNetDeepFashion model.

        Args:
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, num_channels, height, width).

        Returns:
        ----------
        A tuple containing the output tensor of the ResNet backbone and the
        output tensor of the classification head.
        """
        return self.resnet(x)
    
if __name__ == '__main__':
    model = ResNetDeepFashion(backbone='resnet_18', cls_head_type='linear', embedding_sz=512)
    
            