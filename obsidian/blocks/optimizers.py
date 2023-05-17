import sys
import inspect
import torch.optim
from obsidian.core.registry import Registry
from obsidian.core import get_members

OPTIMIZERS = Registry()

for k, v in get_members('torch.optim', disregard=['Optimizer']):
    OPTIMIZERS.registry[k] = v
