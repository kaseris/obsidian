import logging

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

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
                 criterion: nn.Module,
                 device,
                 n_epochs: int,
                 experiment_tracker: ExperimentTracker = None):
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
            inputs, targets = batch['img'], batch['category']
            logging.info(f'Epoch: [{self.epoch + 1}/{self.n_epochs}] Step: [{batch_idx}/{len(self.train_loader)}]')
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logging.debug(f'Inputs shape: {inputs.shape}, Targets shape: {targets.shape}')
            self.optimizer.zero_grad()
            if targets.ndim > 1:
                targets = targets.squeeze()
                logging.debug(f'Targets shape: {targets.shape}')
            outputs, _, _ = self.model(inputs, targets)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            logging.debug(f'Predicted: {predicted}')
            correct = predicted.eq(targets).sum().item()
            train_correct += correct
            logging.debug(f'Correct preds: {correct}')
            total += targets.size(0)
            
            if self.experiment_tracker is not None:
                self.experiment_tracker.log_metrics({'train_loss': loss.item(),
                                                     'train_acc': train_correct / total,
                                                     'epoch': epoch,
                                                     'step': batch_idx})
            
        return train_loss / len(self.train_loader), train_correct / total
    
    def validate_epoch(self, epoch):
        """
        Validate the model
        
        Args:
            epoch (int): current epoch number
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                inputs, targets = batch['img'], batch['category']
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                
                if self.experiment_tracker is not None:
                    self.experiment_tracker.log_metrics({'val_loss': loss.item(),
                                                         'val_acc': val_correct / total,
                                                         'epoch': epoch,
                                                         'step': batch_idx})
                
        return val_loss / len(self.val_loader), val_correct / total
    
    def train(self):
        """
        Train the model
        
        Args:
            epochs (int): number of epochs to train the model
        """
        self.model.to(self.device)
        self.criterion.to(self.device)
        
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)
            
            if self.experiment_tracker is not None:
                self.experiment_tracker.log_epoch(train_loss, train_acc, val_loss, val_acc, epoch)
