import sys
import inspect
import torch.optim
from obsidian.core.registry import Registry
from obsidian.core import get_members

OPTIMIZERS = Registry()

for k, v in get_members('torch.optim', disregard=['Optimizer']):
    OPTIMIZERS.registry[k] = v


def get_optimizer(optimizer_name, **kwargs):
    optimizer_cls = OPTIMIZERS[optimizer_name]
    return optimizer_cls(**kwargs)
