import sys
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection as detection

from dataset import DATASETS
from registry import Registry

DETECTORS = Registry()
DETECTOR_WEIGHTS = Registry()
BACKBONES =  Registry()
BACKBONE_WEIGHTS = Registry()

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
    weights_backbone = getattr(BACKBONE_WEIGHTS[weights_backbone['name']], weights_backbone['checkpoint'])
    num_classes = detector_cfg['cfg'].pop('num_classes')
    detector = DETECTORS[detector_cfg['name']](weights=weights,
                                               weights_backbone=weights_backbone)
    return detector
