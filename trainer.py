import logging

from typing import List, Optional, Tuple, Dict, Any


import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from callbacks import CallbackList, Callback
from model import OBSModule
from registry import Registry
from trackers import ExperimentTracker

TRAINERS = Registry()


@TRAINERS.register('simple-trainer')
class Trainer:
    def __init__(self,
                 model: OBSModule,
                 train_loader: data.DataLoader,
                 val_loader: data.DataLoader,
                 optimizer: optim.Optimizer,
                 device,
                 n_epochs: int,
                 criterion: nn.Module = None,
                 callbacks: Optional[List[Callback]] = None,
                 scaler: torch.cuda.amp.GradScaler = None,
                 experiment_tracker: ExperimentTracker = None,
                 **kwargs):
        """
        A class to train and validate a PyTorch model

        Args:
            model (nn.Module): PyTorch model to train and validate
            train_loader (DataLoader): PyTorch DataLoader containing the training dataset
            val_loader (DataLoader): PyTorch DataLoader containing the validation dataset
            optimizer (Optimizer): PyTorch optimizer for training the model
            criterion (Loss): PyTorch loss function for computing the training loss
            device (str): device to run the computation on (e.g. 'cpu', 'cuda')
            experiment_tracker (ExperimentTracker, optional): experiment tracker for logging training metrics (e.g. wandb)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.experiment_tracker = experiment_tracker
        self.n_epochs = n_epochs
        self.epoch = 0
        self.scaler = scaler
        self.experiment_tracker = experiment_tracker
        if callbacks is not None:
            self.callback_list = CallbackList(callbacks)

    def train_epoch(self, epoch):
        """
        Train one epoch

        Args:
            epoch (int): current epoch number
        """
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0

        for batch_idx, batch in enumerate(self.train_loader):
            x, targets = batch
            res = self.model.training_step(x, targets, device=self.device,
                                           optimizer=self.optimizer,
                                           scaler=self.scaler,
                                           lr_scheduler=None)
            print(res)
            if self.experiment_tracker is not None:
                pass

        return res

    def validate_epoch(self, epoch):
        """
        Validate the model

        Args:
            epoch (int): current epoch number
        """
        for batch_idx, batch in enumerate(self.val_loader):
            x, targets = batch
            res = self.model.validation_step(x, targets, device=self.device)
        return res

    def train(self):
        """
        Train the model

        Args:
            epochs (int): number of epochs to train the model
        """
        self.model.to(self.device)
        # self.callback_list.on_train_begin()
        for epoch in range(self.n_epochs):
            # self.callback_list.on_epoch_begin(epoch)
            self.epoch = epoch
            res = self.train_epoch(epoch)
            result = self.validate_epoch(epoch)
            # self.callback_list.on_epoch_end(epoch,
            #                                 result=result, summarize=True)

            if self.experiment_tracker is not None:
                pass
