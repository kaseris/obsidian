import os.path as osp
import json

from model import MODELS
from dataset import DATASETS


def build_model(cfg):
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
        