import os

from abc import ABC, abstractmethod
import logging
import torch.nn as nn

from core.registry import Registry

TRACKERS = Registry()


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment trackers. Defines the methods that must be implemented by subclasses.

    Args:
        project_name (str): Name of the project.
        experiment_name (str): Name of the experiment.

    Attributes:
        project_name (str): Name of the project.
        experiment_name (str): Name of the experiment.
    """

    def __init__(self, project_name: str, experiment_name: str):
        self.project_name = project_name
        self.experiment_name = experiment_name

    @abstractmethod
    def log_parameters(self, params: dict):
        """
        Log hyperparameters of the experiment.

        Args:
            params (dict): A dictionary of hyperparameters and their values.
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int = None):
        """
        Log metrics of the experiment.

        Args:
            metrics (dict): A dictionary of metrics and their values.
            step (int, optional): The step number for which to log the metrics.
        """
        pass

    @abstractmethod
    def log_artifact(self, path: str):
        """
        Log an artifact (e.g. a model file) for the experiment.

        Args:
            path (str): The path to the artifact file.
        """
        pass


@TRACKERS.register('wandb')
class WandbExperimentTracker(ExperimentTracker):
    """
    Implementation of an experiment tracker using the Weights & Biases (wandb) library.

    Args:
        project_name (str): Name of the project.
        experiment_name (str): Name of the experiment.
        entity (str, optional): Name of the entity that owns the project (e.g. a team name).
        run_name (str, optional): Name of the run (defaults to a unique ID).
        tags (list of str, optional): List of tags for the experiment.
        config (dict, optional): Dictionary of hyperparameters and their values.
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        entity: str = None,
        run_name: str = None,
        tags: list = None,
        config: dict = None
    ):
        logging.debug(f'Creating {__class__.__name__} experiment tracker.')
        super().__init__(project_name, experiment_name)
        import wandb
        logging.debug('Attempting log in.')
        wandb.login(key=os.environ['WANDB_API_KEY'])
        wandb.init(
            project=self.project_name,
            name=run_name,
            entity=entity,
            tags=tags,
            config=config,
            reinit=True
        )
        self.wandb = wandb

    def log_parameters(self, params: dict):
        """
        Log hyperparameters of the experiment.

        Args:
            params (dict): A dictionary of hyperparameters and their values.
        """
        self.wandb.config.update(params)

    def log_metrics(self, metrics: dict):
        """
        Log metrics of the experiment.

        Args:
            metrics (dict): A dictionary of metrics and their values.
            step (int, optional): The step number for which to log the metrics.
        """
        self.wandb.log(metrics)

    def log_artifact(self, path: str):
        """
        Log an artifact (e.g. a model file) for the experiment.

        Args:
            path (str): The path to the artifact file.
        """
        self.wandb.save(path)

    def watch(self, model: nn.Module):
        self.wandb.watch(model)
