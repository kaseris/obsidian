import sys
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from obsidian.core.registry import Registry

LOSSES = Registry()

"""
Build a dictionary of loss classes available in PyTorch's 'nn' module.

The dictionary's keys are strings that correspond to the names of the loss classes, and the values are the
loss classes themselves.

Returns:
    dict: A dictionary mapping loss class names to their corresponding classes.
"""
for member, obj in inspect.getmembers(sys.modules['torch.nn.modules.loss']):
    if member in ['Tensor', 'Module']:
        continue
    if inspect.isclass(obj):
        LOSSES.registry[member] = obj


@LOSSES.register('BatchHardTripletLoss')
class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    @staticmethod
    def get_anchor_positive_triplet_mask(target):
        """
        Given a target tensor, returns a boolean mask tensor indicating which embeddings in the batch are anchor and
        positive embeddings.

        Args:
            target (torch.Tensor): A tensor containing target labels for the batch.

        Returns:
            torch.Tensor: A boolean mask tensor with the same shape as `target` indicating which embeddings are anchor
            and positive embeddings.
        """
        mask = torch.eq(target.unsqueeze(0), target.unsqueeze(1))
        mask.fill_diagonal_(False)
        return mask

    @staticmethod
    def get_anchor_negative_triplet_mask(target):
        """
        Given a target tensor, returns a boolean mask tensor indicating which embeddings in the batch are anchor and
        negative embeddings.

        Args:
            target (torch.Tensor): A tensor containing target labels for the batch.

        Returns:
            torch.Tensor: A boolean mask tensor with the same shape as `target` indicating which embeddings are anchor
            and negative embeddings.
        """
        labels_equal = torch.eq(target.unsqueeze(0), target.unsqueeze(1))
        mask = ~ labels_equal
        return mask

    def forward(self, x, target):
        """
        Computes the batch-hard triplet loss for the given batch of embeddings.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, embedding_dim) containing the embeddings for the batch.
            target (torch.Tensor): A tensor of shape (batch_size,) containing the target labels for the batch.

        Returns:
            torch.Tensor: The triplet loss for the given batch of embeddings.
        """
        pairwise_dist = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)

        mask_anchor_positive = self.get_anchor_positive_triplet_mask(target)
        anchor_positive_dist = mask_anchor_positive.float() * pairwise_dist
        hardest_positive_dist = anchor_positive_dist.max(1, True)[0]

        mask_anchor_negative = self.get_anchor_negative_triplet_mask(target)
        # make positive and anchor to be exclusive through maximizing the dist
        max_anchor_negative_dist = pairwise_dist.max(1, True)[0]
        anchor_negative_dist = pairwise_dist + \
            max_anchor_negative_dist * (1.0 - mask_anchor_negative.float())
        hardest_negative_dist = anchor_negative_dist.min(1, True)[0]

        loss = (F.relu(hardest_positive_dist -
                hardest_negative_dist + self.margin))
        return loss.mean()


if __name__ == '__main__':
    print(LOSSES.registry.keys())
