import os
import sys
import inspect
import json
import logging
import os.path as osp
from collections.abc import Mapping, MutableMapping
from itertools import chain
from operator import add

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from datasets.dataset import DATASETS, TRANSFORMS
from blocks.detection.detection import DETECTORS
from core.fileclient import read_file
from blocks.losses.losses import LOSSES
from blocks.classification.model import MODELS
from core.trackers import TRACKERS
from core.trainer import TRAINERS, Trainer
from obsidian.core.utils import prepare_data, import_registry


_FLAG_FIRST = object()
NAME_2_TASK = {
    'detection': 'FashionDetector',
    'classification': ['model', 'MODELS']
}


def flattenDict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flattenDict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


"""
Build a dictionary of optimizer classes available in PyTorch's 'optim' module.

The dictionary's keys are strings that correspond to the names of the optimizer classes, and the values are the
optimizer classes themselves.

Returns:
    dict: A dictionary mapping optimizer class names to their corresponding classes.
"""
OPTIMIZERS = dict()
for member, obj in inspect.getmembers(sys.modules['torch.optim']):
    if inspect.isclass(obj):
        OPTIMIZERS[member] = obj


def build_model(cfg) -> nn.Module:
    """
    Builds a model from a dictionary read from a JSON file.

    Args:
        cfg (dict): A dictionary containing the model configuration.

    Returns:
        nn.Module: A PyTorch model instantiated from the configuration.

    Raises:
        KeyError: If the configuration dictionary does not have a 'type' key.
        ValueError: If the 'type' key in the configuration does not correspond
            to a registered model type in the MODELS registry.

    Example:
    >>> with open('configs/base_model.json', 'r') as f:
    >>>     data = json.load(f)
    >>> model = build_model(data['model'])
    >>> print(model)
    """
    # TODO: Automatically infer whether the model is a detector or a classifier, etc. Done
    # TODO: Separate classifiers and detectors.
    task = cfg['task']
    model_cfg = cfg['model']
    model_cfg = model_cfg['cfg']
    model_type = cfg['model']['name']

    logging.debug(
        f'The task is {task} and the requested model is {model_type}.')
    module_name = NAME_2_TASK[task]
    # registry = import_registry(
    #     module_name=module_name, registry_name=registry_name)
    # logging.debug(
    #     f'The model registry has the keys {registry.registry.keys()}.')
    # if model_type not in MODELS.registry.keys():
    #     raise ValueError(f'Unsupported model type: `{model_type}`. '
    #                      f'Must be one of \n{list(MODELS.registry.keys())}')
    # module = registry.registry[model_type](**model_cfg)
    model = MODELS[module_name](**cfg['model'])
    return model


def build_dataset(data: dict) -> torch.utils.data.Dataset:
    """
    Builds a PyTorch dataset based on the configuration specified in `cfg`.

    Args:
        cfg (dict): A dictionary containing the dataset configuration information.

    Returns:
        torch.utils.data.Dataset: The PyTorch dataset built according to the configuration.

    Raises:
        ValueError: If `dataset_type` is not found in the DATASETS registry.

    Example:
    ```
    if __name__ == '__main__':
    with open(osp.join('configs', 'dataset_config.json'), 'r') as f:
        data = json.load(f)
    dataset = build_dataset(data['dataset'])
    print(dataset)
    ```
    """
    # Find all the dataset related keys
    dataset_keys = list(filter(lambda x: 'dataset' in x, data.keys()))
    ret = dict()
    logging.info(f'Building datasets.')
    logging.debug(f'Dataset keys: {", ".join(k for k in dataset_keys)}')
    for key in dataset_keys:
        logging.info(f'Building dataset based on `{key}` dataset')
        dataset_cfg = data[key]['cfg']
        dataset_type = data[key]['name']
        logging.debug(f'Config:\n {dataset_cfg}')

        if dataset_type not in DATASETS.registry.keys():
            raise ValueError(f'Unsupported model type: `{dataset_type}`. '
                             f'Must be one of \n{list(DATASETS.registry.keys())}')
        dataset_cfg['transforms'] = TRANSFORMS[dataset_cfg['transforms']['name']](
            **dataset_cfg['transforms']['cfg'])
        # if key == 'train_dataset':
        #     logging.debug('Calling `prepare_data` function.')
        #     splits, annotations = prepare_data()
        #     logging.info('Data preparation done.')
        # dataset_cfg['split_info'] = splits
        # dataset_cfg['garment_annotations'] = annotations
        ret[key] = DATASETS[dataset_type](
            root=os.environ['DATASET_DIR'], **dataset_cfg)

    return ret


def build_optimizer(cfg: dict,
                    model: nn.Module) -> optim.Optimizer:
    """
    Builds a PyTorch optimizer object using the configuration provided in the `cfg` dictionary.

    Args:
        cfg (dict): A dictionary containing the configuration for the optimizer. It should have the following keys:
            - "type": A string indicating the type of optimizer to use. This should match the name of one of the classes
                      available in the PyTorch `torch.optim` module.
            - "optimizer_cfg": A dictionary containing any additional configuration options to pass to the optimizer
                               constructor.
            - "params": The modules whose params are to be trained. If set to null in the config file (None in Python
                        format), the whole model's parameters will be added to the optimizer.
        model (nn.Module): A model, whose designated params are to be added to the parameter list of the optimizer.

    Returns:
        A PyTorch optimizer object.

    Raises:
        ValueError: If the optimizer type specified in the `cfg` dictionary is not supported. This can happen if the
                    name provided is not a valid optimizer class in the `torch.optim` module.

    Example:
        Here's an example of how to use the `build_optimizer` function to create an Adam optimizer:

        >>> cfg = {"type": "Adam", "params": None, "optimizer_cfg": {"lr": 0.001}}
        >>> optimizer = build_optimizer(cfg)
        >>> print(optimizer)
        Adam (
        Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.999)
            eps: 1e-08
            lr: 0.001
            weight_decay: 0
        )
    """
    logging.info('Building optimizer.')
    optimizer_type = cfg['name']
    optimizer_cfg = cfg['cfg']
    logging.debug(f'Config:\n{optimizer_cfg}')
    # logging.debug(f'Selected modules:\n{cfg["params"]}')
    trainable_params = []
    if cfg['params'] is not None:
        for name, module in model.named_modules():
            if name in cfg['params']:
                logging.debug(f'Module: {module}')
                optimizer_cfg['params'] = module.parameters()
    else:
        optimizer_cfg["params"] = model.params
    if optimizer_type not in OPTIMIZERS.keys():
        raise ValueError(f'Unsupported model type: `{optimizer_type}`. '
                         f'Must be one of \n{list(OPTIMIZERS.keys())}')

    return OPTIMIZERS[optimizer_type](**optimizer_cfg)


def build_trainer(cfg_file) -> Trainer:
    data = read_file(cfg_file)
    flat_data = flattenDict(data)
    logging.debug(f'Flattened data: {flat_data}')
    model_cfg = data['model']
    logging.debug(f'Model configuration: {model_cfg}')
    logging.debug(f'Task: {data["task"]}')
    logging.info('Building model...')
    model = build_model(cfg=data)
    logging.info('Building datasets...')
    dataset_dict = build_dataset(data=data)

    logging.info('Creating optimizers.')
    optimizer_cfg = data['optimizer']
    logging.debug(f'Optimizer configuration: {optimizer_cfg}')
    opt = build_optimizer(cfg=optimizer_cfg, model=model)

    # Needs to be implemented under the factory design pattern
    logging.info('Creating data loaders.')
    train_loader = DataLoader(
        dataset=dataset_dict['dataset_train'], **data['train_loader']['loader_cfg'],
        collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(
        dataset=dataset_dict['dataset_val'], **data['val_loader']['loader_cfg'],
        collate_fn=lambda x: tuple(zip(*x)))
    # ========================================================

    trainer_type = data['trainer']['type']
    logging.debug(f'Trainer type: {trainer_type}')
    trainer_cfg = data['trainer']['trainer_cfg']
    logging.debug(f'Trainer configuration: {trainer_cfg}')
    # criterion = LOSSES[data['trainer']['trainer_cfg']['criterion']]()
    criterion = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.debug(f'Device: {device}')

    tracker_cfg = data['trainer']['trainer_cfg']['tracker']
    tracker = TRACKERS[tracker_cfg](project_name='fashion-detection',
                                    experiment_name='base-pre-train')
    tracker.log_parameters(flat_data)

    trainer_cfg = {
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'optimizer': opt,
        'criterion': criterion,
        'device': device,
        'experiment_tracker': tracker,
        'n_epochs': data['trainer']['trainer_cfg']['n_epochs']
    }

    return TRAINERS[trainer_type](**trainer_cfg)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cfg = read_file('configs/detection/base.yaml')
    build_model(cfg=cfg)
