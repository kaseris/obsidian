import json
import logging
import os.path as osp

import torch
import torch.nn as nn

from dataset import DATASETS
from model import MODELS
from utils import prepare_data


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
    model_cfg = cfg['model_cfg']
    model_type = cfg['type']

    if model_type not in MODELS.registry.keys():
        raise ValueError(f'Unsupported model type: `{model_type}`. '
                         f'Must be one of \n{list(MODELS.registry.keys())}')

    return MODELS[model_type](**model_cfg)

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
    logging.debug(f'Dataset keys: {" ,".join(k for k in dataset_keys)}')
    for key in dataset_keys:
        logging.info(f'Building dataset based on `{key}` dataset')
        dataset_cfg = data[key]['dataset_cfg']
        dataset_type = data[key]['type']
        logging.debug(f'Config:\n {dataset_cfg}')
        
        if dataset_type not in DATASETS.registry.keys():
            raise ValueError(f'Unsupported model type: `{dataset_type}`. '
                            f'Must be one of \n{list(DATASETS.registry.keys())}')
        
        if key == 'train_dataset':
            logging.debug('Calling `prepare_data` function.')
            splits, annotations = prepare_data()
            logging.info('Data preparation done.')
        dataset_cfg['split_info'] = splits
        dataset_cfg['garment_annotations'] = annotations
        ret[key] = DATASETS[dataset_type](**dataset_cfg)
        
    return ret

if __name__ == '__main__':
    with open(osp.join('configs', 'base_model.json'), 'r') as f:
        data = json.load(f)
    logging.basicConfig(level=logging.DEBUG)
    datasets = build_dataset(data=data)
    print(datasets)
    