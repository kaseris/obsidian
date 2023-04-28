from abc import ABC, abstractmethod

import torch.nn as nn
import torch.utils.data

from coco.coco_eval import CocoEvaluator
from coco.coco_utils import get_coco_api_from_dataset
from coco.engine import _get_iou_types

from registry import Registry


CALLBACKS = Registry()


class Callback(ABC):
    """
    A base class for defining callbacks during model training.

    Attributes:
        None

    Methods:
        `on_train_begin(model, **kwargs)`: called at the beginning of the training process.

        `on_train_end(model, **kwargs)`: called at the end of the training process.

        `on_epoch_begin(model, epoch, **kwargs)`: called at the beginning of each epoch.

        `on_epoch_end(model, epoch, **kwargs)`: called at the end of each epoch.
    """
    @abstractmethod
    def on_train_begin(self, **kwargs):
        pass
    
    @abstractmethod
    def on_train_end(self, **kwargs):
        pass
    
    @abstractmethod
    def on_epoch_begin(self, epoch, **kwargs):
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch, **kwargs):
        pass

@CALLBACKS.register('CocoEvaluationCallback')
class CocoEvaluationCallback(Callback):
    """
    A callback to perform COCO evaluation on a given dataset after each epoch of training.
    
    Args:
        `dataset (torch.utils.data.Dataset)`: The dataset on which to perform COCO evaluation.
        
        `model (nn.Module)`: The neural network model to use for inference.
        
    Attributes:
        `names (List[str])`: A list of metric names for which the evaluation results will be reported.
        
    Methods:
        `on_train_begin(**kwargs)`: A callback function that is called at the beginning of the training process.

        `on_train_end(**kwargs)`: A callback function that is called at the end of the training process.

        `on_epoch_begin(epoch, **kwargs)`: A callback function that is called at the beginning of each epoch.

        `on_epoch_end(epoch, **kwargs)`: A callback function that is called at the end of each epoch.
        
    Example usage:
        >>> dataset = MyDataset()
        >>> model = MyModel()
        >>> evaluator = CocoEvaluationCallback(dataset, model)
        >>> callback_list = CallbackList([evaluator])
        >>> trainer = MyTrainer(callback_list=callback_list)
        >>> trainer.train()
    """
    names = [
    'AP@IoU[0.50:0.95]',
    'AP@IoU[0.50]',
    'AP@IoU[0.75]',
    'AP@IoU[0.50:0.95]_small',
    'AP@IoU[0.50:0.95]_medium',
    'AP@IoU[0.50:0.95]_large',
    'AR@IoU[0.50:0.95]_maxdets1_all',
    'AR@IoU[0.50:0.95]_maxdets10_all',
    'AR@IoU[0.50:0.95]_maxdets100_all',
    'AR@IoU[0.50:0.95]_maxdets100_small',
    'AR@IoU[0.50:0.95]_maxdets100_medium',
    'AR@IoU[0.50:0.95]_maxdets100_large'
    ]

    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 model: nn.Module) -> None:
        super().__init__()
        self.dataset = dataset
        self.model = model

        self.coco_evaluator = None
        self.coco = None
        
    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_begin(self, epoch, **kwargs):
        self.coco = get_coco_api_from_dataset(self.dataset)
        iou_types = _get_iou_types(self.model)
        self.coco_evaluator = CocoEvaluator(self.coco, iou_types)

    def on_epoch_end(self, epoch, **kwargs):
        result = kwargs.get('result')
        summarize = kwargs.get('summarize')
        self.coco_evaluator.update(result)
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        if summarize:
            self.coco_evaluator.summarize()
        
        stats = {}
        for iou_type, coco_eval in self.coco_evaluator.coco_eval.items():
            _stats = coco_eval.stats.tolist()
            for idx, name in enumerate(CocoEvaluationCallback.names):
                stats[iou_type+ + '_' + name] = _stats[idx]


class CallbackList:
    """
    A class that manages a list of callback objects and calls their methods during model training.

    Attributes:
        callbacks (list): a list of callback objects to manage.

    Methods:
        `on_train_begin(model, **kwargs)`: called at the beginning of the training process.

        `on_train_end(model, **kwargs)`: called at the end of the training process.

        `on_epoch_begin(model, epoch, **kwargs)`: called at the beginning of each epoch.

        `on_epoch_end(model, epoch, **kwargs)`: called at the end of each epoch.
    """
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_train_begin(self, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(**kwargs)

    def on_train_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)

    def on_epoch_begin(self, epoch, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, **kwargs)

    def on_epoch_end(self, epoch, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, **kwargs)


if __name__  == '__main__':
    pass
    