from abc import ABC, abstractmethod

import torch.nn as nn


class OBSModule(ABC, nn.Module):
    """
    Abstract base class for defining modules that implement online bootstrapping (OBS).
    Child classes should implement the abstract methods `forward`, `training_step`, and `validation_step`.

    Attributes:
    -----------
    None

    Methods:
    --------
    forward(x):
        Forward pass of the module.
        Arguments:
        ----------
        x: torch.Tensor
            Input tensor to the module.
        Returns:
        --------
        output: torch.Tensor
            Output tensor of the module.

    training_step(*args, **kwargs):
        Abstract method to define the training step for the OBS module.
        Arguments:
        ----------
        *args, **kwargs: Any
            Additional arguments to be passed to the training step.
        Returns:
        --------
        loss: torch.Tensor
            Loss tensor for the training step.

    validation_step(*args, **kwargs):
        Abstract method to define the validation step for the OBS module.
        Arguments:
        ----------
        *args, **kwargs: Any
            Additional arguments to be passed to the validation step.
        Returns:
        --------
        output: torch.Tensor
            Output tensor of the validation step.
    """
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def training_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def validation_step(self, *args, **kwargs):
        pass
