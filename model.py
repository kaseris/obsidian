import time
from abc import ABC, abstractmethod
import logging
import sys
from typing import Union, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

import config

from coco.coco_eval import CocoEvaluator
from coco.utils import reduce_dict
from detection import build_detector
from pooling import DESCRIPTORS, POOLING
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


@BACKBONES.register('resnet_152')
def resnet_152():
    """
    Returns a ResNet152 instance pretrained on ImageNet.
    """
    return models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)


@BACKBONES.register('vit_b_16')
def vit_b_16():
    """
    Returns a vision transformer (ViT) instance pretrained on ImageNet.
    """
    return models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)


@CLS_HEADS.register('simple')
class ClassificationHead(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(ClassificationHead, self).__init__()
        if kwargs.get('embedding_sz'):
            setattr(self, 'embedding_sz', kwargs.get('embedding_sz'))
        self.cls_head = nn.Linear(in_features=kwargs['fan_in'],
                                  out_features=kwargs['n_classes'])

    def forward(self, x: torch.Tensor):
        if x.ndim > 2:
            x = x.squeeze()
        embedding = x
        out_cls = self.cls_head(embedding)
        return out_cls, embedding

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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

    def __init__(self, **kwargs) -> None:
        super(LinearClassificationHead, self).__init__()
        self.embedding = nn.Linear(in_features=kwargs['fan_in'],
                                   out_features=kwargs['embedding_sz'])
        self.act = nn.ReLU()
        self.cls_head = nn.Linear(in_features=kwargs['embedding_sz'],
                                  out_features=kwargs['n_classes'])

    def forward(self, x):
        embedding = self.embedding(x.squeeze())
        out_cls = self.cls_head(embedding)
        return out_cls, embedding

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


@CLS_HEADS.register('cgd_head')
class CombinedGlobalDescriptorClassHead(nn.Module):
    """
    A PyTorch module that combines global descriptors and a classification head
    to classify image data.

    Args:
        fan_in (int): Number of channels in the input feature map.
        feat_dim (int): Dimensionality of the output feature map of the global descriptors.
        n_classes (int): Number of classes for classification.
        gd_config (str, optional): String specifying the configuration of the global
            descriptors to use. This should be a string composed of one or more of the
            following characters: 'S' (for a descriptor using sum-pooling), 'M' (for
            a descriptor using max-pooling), and 'G' (for a descriptor using generalized
            mean pooling). Defaults to 's' (a single descriptor using sum-pooling).

    Attributes:
        gd (CombinedGlobalDescriptor): A CombinedGlobalDescriptor module that computes
            one or more global descriptors from the input feature map.
        bn (nn.BatchNorm2d): A BatchNorm2d layer that normalizes the output of the
            first global descriptor.
        cls (nn.Linear): A linear layer that computes the logits for classification.

    Methods:
        forward(x): Computes the logits for classification from the input feature map.
        init_weights(): Initializes the weights of the batch normalization and linear layers.

    Raises:
        AssertionError: If `gd_config` is not a valid string specifying the global descriptor
            configuration, or if `feat_dim` is not divisible by the number of global descriptors.
    """

    def __init__(self,
                 **kwargs) -> None:
        super(CombinedGlobalDescriptorClassHead, self).__init__()
        self.gd = DESCRIPTORS['config_descriptor'](fan_in=kwargs['fan_in'],
                                                   gd_config=kwargs['gd_config'],
                                                   feat_dim=kwargs['feat_dim'])
        self.bn = nn.BatchNorm2d(num_features=kwargs['fan_in'])
        self.cls = nn.Linear(
            in_features=kwargs['fan_in'], out_features=kwargs['n_classes'], bias=True)

    def forward(self, x: torch.Tensor):
        gd, first_gd = self.gd(x)
        out = self.bn(first_gd)
        out = self.cls(out)
        return out, gd

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class OBSModule(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def training_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def validation_step(self, *args, **kwargs):
        pass


@MODELS.register('ResNetDeepFashion')
class ResNetDeepFashion(OBSModule):
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
                 **kwargs):
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
        self.cls_head_type = kwargs['cls_head_type']
        self.backbone = BACKBONES[kwargs['backbone']]()
        self.cls_head = None
        self.optimizer = None
        self.cls_head_config = kwargs['cls_head_config']
        self._prepare_model()

    def _prepare_model(self):
        """
        Prepare the ResNetDeepFashion model by modifying the final layer.
        """
        num_features = list(self.backbone.children())[-1].in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.cls_head_config['fan_in'] = num_features
        self.cls_head = CLS_HEADS[self.cls_head_type](**self.cls_head_config)

    def freeze_weights(self):
        """
        Freezes the weights of the `backbone`'s layers.
        """
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        """
        Unfreezes the weights of the `backbone`'s layers.
        """
        for name, param in self.backbone.named_parameters():
            param.requires_grad = True

    def configure_optimizer(self, optimizer: torch.optim.Optimizer):
        """
        Configure the optimizer used during training.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to use for training.

        Returns:
            None

        Examples:
        ```
            >>> model = ResNetDeepFashion()
            >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            >>> model.configure_optimizer(optimizer)
        ```
        """
        self.optimizer = optimizer

    def training_step(self, x: torch.Tensor,
                      targets: torch.Tensor):
        """
            Performs a single training step on a batch of input data and targets.

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, num_channels, height, width).
            targets (torch.Tensor): Target tensor of shape (batch_size) containing integer class labels.

        Returns:
            dict: A dictionary containing the training accuracy and loss for the batch.
                - train_acc (float): The training accuracy for the batch as a percentage.
                - train_loss (float): The training loss for the batch.

        Raises:
            ValueError: If the `targets` tensor has more than one dimension.

        This method first checks if the `targets` tensor has more than one dimension, and if so, it squeezes it down to one dimension.
        It then passes the input data and targets to the model to obtain logits and loss, and computes the training accuracy as the percentage of correct predictions.
        The model's optimizer is then zeroed, the loss is backpropagated through the network, and the optimizer is stepped forward.
        The method returns a dictionary containing the training accuracy and loss for the batch.
        """
        # Cross-entropy loss requires 0-D or 1-D target inputs.
        if targets.ndim > 1:
            targets = targets.squeeze()
        logits, _, loss = self(x, targets)
        logits_cpu = logits.to('cpu')
        targets_cpu = targets.to('cpu')
        predictions = torch.argmax(logits_cpu, dim=1)
        train_accuracy = torch.sum(
            predictions == targets_cpu) / len(targets_cpu)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'train_acc': train_accuracy.item(),
                'train_loss': loss.item()}

    @torch.no_grad()
    def validation_step(self, x_val: torch.Tensor,
                        targets_val: torch.Tensor):
        """
        Performs a single validation step on a batch of input data `x_val` with associated targets `targets_val`. 
        Computes the cross-entropy loss and accuracy for the given batch.

        Args:
            x_val (torch.Tensor): A tensor of input images of shape (batch_size, channels, height, width).
            targets_val (torch.Tensor): A tensor of integer labels of shape (batch_size,) indicating the ground-truth 
                                        class for each input image.

        Returns:
            A dictionary with the following keys:
                - 'val_acc' (float): The validation accuracy for the given batch.
                - 'val_loss' (float): The validation loss for the given batch.
        """
        if targets_val.ndim > 1:
            targets_val = targets_val.squeeze()
        logits, _, val_loss = self(x_val, targets_val)
        val_loss = val_loss.to('cpu').item()
        logits_cpu = logits.to('cpu')
        targets_cpu = targets_val.to('cpu')
        predictions = torch.argmax(logits_cpu, dim=1)
        val_accuracy = torch.sum(predictions == targets_cpu) / len(targets_cpu)
        return {'val_acc': val_accuracy.item(),
                'val_loss': val_loss}

    def forward(self, x: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the ResNetDeepFashion model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_channels, height, width).
            targets (torch.Tensor, optional): The target tensor of shape (batch_size) containing the ground-truth
                class indices. If not None, the function will return the output tensor of the ResNet backbone, the 
                output tensor of the classification head and the classification loss.

        Returns:
            A tuple containing the embeddings tensor of the Classification head layer, the output tensor of the classification head
            and the classification loss if targets is not None, otherwise only the embedings of the `cls_head`
            and the output tensor of the classification head. The loss is set to `None` in this case.
        """
        with torch.no_grad():
            out = self.backbone(x)
        preds, embeddings = self.cls_head(out)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(preds, target=targets)
            return preds, embeddings, loss
        return preds, embeddings, loss


@MODELS.register('FashionDetector')
class FashionDetector(OBSModule):
    """
    Class for training, evaluating and validating a fashion object detection model.

    Attributes:
        module (nn.Module): the object detection model.

    Methods:
        __init__(self, **kwargs): initializes the FashionDetector instance.

        forward(self): a placeholder function for model inference.

        validation_step(self, images, targets, device): performs a single validation step.

        training_step(self, x, targets, device, optimizer, scaler, lr_scheduler): performs a single training step.

        params(self): returns the model's trainable parameters.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        logging.debug(f'Initializing FashionDetector with kwargs: {kwargs}')
        if 'debug' in kwargs['cfg']:
            setattr(self, 'debug', kwargs['cfg']['debug'])
        self.module = build_detector(**kwargs)

    def forward(self):
        pass

    @torch.inference_mode()
    def validation_step(self,
                        images,
                        targets,
                        device):
        """
        Performs a single validation step.

        Args:
            images (list of torch.Tensor): the input images.

            targets (list of torch.Tensor): the ground truth targets for the given images.

            device (str): the device to perform inference on.

        Returns:
            res (dict): a dictionary with the model's outputs for each input image.
        """
        self.module.eval()
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()
        outputs = self.module(images)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        logging.debug(f'Model inference time: {model_time}')
        res = {target["image_id"].item(): output for target,
               output in zip(targets, outputs)}
        return res

    def training_step(self,
                      x: torch.Tensor,
                      targets,
                      device,
                      optimizer: torch.optim.Optimizer,
                      scaler: torch.cuda.amp.grad_scaler.GradScaler = None,
                      lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None):
        """
        Performs a single training step.

        Args:
            x (torch.Tensor): the input images.

            targets (list of torch.Tensor): the ground truth targets for the given images.

            device (str): the device to perform inference on.

            optimizer (torch.optim.Optimizer): the optimizer instance to be used for training.

            scaler (torch.cuda.amp.grad_scaler.GradScaler): the scaler to be used for training.

            lr_scheduler (torch.optim.lr_scheduler.LRScheduler): the learning rate scheduler to be used for training.

        Returns:
            loss_dict_reduced (dict): a dictionary with the model's reduced losses over all GPUs.
        """
        logging.debug(f'Training step with device: {device}')
        logging.debug(f'Training step with scaler: {scaler}')
        logging.debug(f'Training step with lr_scheduler: {lr_scheduler}')
        self.module.train()
        images = list(img.to(device) for img in x)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=True):
            loss_dict = self.module(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Reduce losses over all GPUs
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        logging.debug(f'Training loss: {losses_reduced.item()}')
        logging.debug(loss_dict_reduced)
        loss_value = losses_reduced.item()

        # Check if we have an unbounded loss value.
        # If we do, we will not perform the backward pass.
        if not torch.isfinite(torch.tensor(loss_value)):
            if hasattr(self, 'debug') and self.debug:
                self.module.eval()
                predictions = self.module(images)
                logging.info('Saving debug data')
                torch.save({'x': x, 'targets': targets,
                           'predictions': predictions}, 'debug.pt')
            logging.error(f'Loss is {loss_value}')
            logging.info(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            logging.debug('Invoking backward with scaler')
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logging.debug('Invoking backward without scaler')
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        return loss_dict_reduced

    @property
    def params(self):
        return [p for p in self.module.parameters() if p.requires_grad]
