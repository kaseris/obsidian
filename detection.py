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

def build_detector(detector_cfg: dict):
    weights = detector_cfg['cfg'].pop('weights')
    weights = getattr(DETECTOR_WEIGHTS[weights['name']], weights['checkpoint'])
    weights_backbone = detector_cfg['cfg'].pop('weights_backbone')
    weights_backbone = getattr(BACKBONE_WEIGHTS[weights_backbone['name']], weights_backbone['checkpoint'])
    num_classes = detector_cfg['cfg'].pop('num_classes')
    detector = DETECTORS[detector_cfg['name']](weights=weights,
                                               weights_backbone=weights_backbone)
    return detector


if __name__ == '__main__':
    import coco.utils
    import coco.engine
    import coco.transforms as T
    # import coco.utils.coco_eval
    import torch.utils
    from torch.optim.lr_scheduler import StepLR

    def get_transform(train):
        transforms = []
        # converts the image, a PIL image, into a PyTorch Tensor
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(dtype=torch.float))
        if train:
            # during training, randomly flip the training images
            # and ground-truth for data augmentation
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    dataset_cfg = {'root': '/home/kaseris/Documents/dev/DeepFashion2mini',
                   'split': 'train'}
    ds = DATASETS['deepfashion2'](transforms=get_transform(train=True), **dataset_cfg)
    dataset_cfg_test = {'root': '/home/kaseris/Documents/dev/DeepFashion2mini',
                        'split': 'validation'}
    ds_test = DATASETS['deepfashion2'](transforms=get_transform(train=False), **dataset_cfg_test)
    import yaml
    with open('configs/detection/base.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    detector = build_detector(cfg['model'])
    # in_features = detector.roi_heads.box_predictor.cls_score.in_features
    # detector.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, 14)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector.to(device)
    params = [p for p in detector.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.00005, weight_decay=0.005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    data_loader = torch.utils.data.DataLoader(ds,
                                              batch_size=2,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=lambda x: tuple(zip(*x)))
    

    data_loader_test = torch.utils.data.DataLoader(ds_test,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   collate_fn=lambda x: tuple(zip(*x)))
    num_epochs = 5

    for epoch in range(num_epochs):
        coco.engine.train_one_epoch(detector, optimizer, data_loader, device, epoch, print_freq=100)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        coco.engine.evaluate(detector, data_loader_test, device=device)