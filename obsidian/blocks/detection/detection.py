import sys
import logging
import time
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection as detection

import obsidian.module as om
from obsidian.coco.utils import reduce_dict
from obsidian.blocks.detection.registry_ import DETECTORS, BACKBONES, DETECTOR_WEIGHTS, BACKBONE_WEIGHTS


for member, obj in inspect.getmembers(sys.modules['torchvision.models.detection']):
    if member in ['Tensor', 'Module']:
        continue
    if (inspect.isclass(obj)) and 'Weights' in member:
        DETECTOR_WEIGHTS.registry[member] = obj

for member, obj in inspect.getmembers(sys.modules['torchvision.models.detection']):
    if member in ['Tensor', 'Module']:
        continue
    if (inspect.isclass(obj)) and 'Weights' not in member:
        DETECTORS.registry[member] = obj
    if inspect.isfunction(obj):
        DETECTORS.registry[member] = obj

for member, obj in inspect.getmembers(sys.modules['torchvision.models']):
    if member in ['Tensor', 'Module']:
        continue
    if (inspect.isclass(obj)) and 'Weights' not in member:
        BACKBONES.registry[member] = obj
    elif inspect.isfunction(obj):
        BACKBONES.registry[member] = obj
    elif inspect.isclass(obj) and 'Weights' in member:
        BACKBONE_WEIGHTS.registry[member] = obj


def build_weights(weight_cfg: dict):
    return getattr(DETECTOR_WEIGHTS[weight_cfg.get('type')],
                   weight_cfg.get('checkpoint'))


def build_backbone(backbone_cfg: dict):
    backbone = BACKBONES[backbone_cfg.get('type')]
    if 'weights' in backbone_cfg:
        weights = BACKBONE_WEIGHTS[backbone_cfg.get('weights')['name']]
        weights = getattr(weights, backbone_cfg.get('weights')['checkpoint'])
    backbone = backbone(weights=weights)

    if 'final_layer' in backbone_cfg:
        final_layer = backbone_cfg['final_layer']
        backbone = nn.Sequential(*list(backbone.children())[:final_layer])
    backbone.out_channels = backbone_cfg.get('out_channels')
    return backbone


def build_detector(**detector_cfg: dict):
    weights = detector_cfg['cfg'].pop('weights')
    weights = getattr(DETECTOR_WEIGHTS[weights['name']], weights['checkpoint'])
    weights_backbone = detector_cfg['cfg'].pop('weights_backbone')
    weights_backbone = getattr(
        BACKBONE_WEIGHTS[weights_backbone['name']], weights_backbone['checkpoint'])
    num_classes = detector_cfg['cfg'].pop('num_classes')
    detector = DETECTORS[detector_cfg['name']](weights=weights,
                                               weights_backbone=weights_backbone)
    return detector


@DETECTORS.register('FashionDetector')
class FashionDetector(om.OBSModule):
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
        if 'debug' in kwargs:
            setattr(self, 'debug', kwargs['debug'])
        self.module = kwargs['detector']

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

    def freeze_layers(self, layer):
        """
        Freezes all layers up to and including the specified layer of a PyTorch model.

        Args:
            layer (int or str): The index or name of the layer up to which to freeze the layers.

        Returns:
            - trainable_params (generator): A generator that yields the trainable parameters of the frozen network.
        """
        if isinstance(layer, int):
            # Layer is an integer representing the index up to which to freeze the layers
            layer_index = layer
        elif isinstance(layer, str):
            # Layer is a string representing the name of the layer up to which to freeze the layers
            layer_index = None
            for i, (name, module) in enumerate(self.module.named_modules()):
                if name == layer:
                    layer_index = i
                    break
            if layer_index is None:
                raise ValueError(f"Layer '{layer}' not found in model")
        else:
            raise ValueError("Layer must be an integer or a string")

        # Set the requires_grad attribute of all layers up to layer_index to False
        for i, param in enumerate(self.module.parameters()):
            if i <= layer_index:
                param.requires_grad = False

        # Return the trainable parameters of the frozen network
        trainable_params = filter(
            lambda p: p.requires_grad, self.module.parameters())

        return trainable_params

    @property
    def params(self):
        return filter(lambda p: p.requires_grad, self.module.parameters())
