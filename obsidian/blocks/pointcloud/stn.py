import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class STN3d(nn.Module):
    """
    A Spatial Transformer Network (STN) module for 3D point clouds.

    This module takes a batch of 3D point clouds as input and applies a series of
    convolutional and fully connected layers to learn a transformation matrix that can
    align the point clouds to a canonical coordinate system. The output of the module is
    a tensor of shape (batch_size, 3, 3), which represents the transformation matrix that
    can align the input point clouds to a canonical coordinate system.

    Args:
        None

    Input:
        x (torch.Tensor): A tensor of shape (batch_size, 3, num_points), where "batch_size"
        is the number of point clouds in the batch, "3" represents the (x, y, z) coordinates
        of each point, and "num_points" is the number of points in each point cloud.

    Output:
        A tensor of shape (batch_size, 3, 3) representing the transformation matrix that can
        align the input point clouds to a canonical coordinate system.

    Attributes:
        conv1 (nn.Conv1d): A convolutional layer that takes input with 3 channels and outputs
        64 channels.
        conv2 (nn.Conv1d): A convolutional layer that takes input with 64 channels and outputs
        128 channels.
        conv3 (nn.Conv1d): A convolutional layer that takes input with 128 channels and outputs
        1024 channels.
        fc1 (nn.Linear): A fully connected layer that takes input with 1024 units and outputs
        512 units.
        fc2 (nn.Linear): A fully connected layer that takes input with 512 units and outputs
        256 units.
        fc3 (nn.Linear): A fully connected layer that takes input with 256 units and outputs
        9 units representing the 3x3 transformation matrix.
        bn1 (nn.BatchNorm1d): A batch normalization layer applied after conv1.
        bn2 (nn.BatchNorm1d): A batch normalization layer applied after conv2.
        bn3 (nn.BatchNorm1d): A batch normalization layer applied after conv3.
        bn4 (nn.BatchNorm1d): A batch normalization layer applied after fc1.
        bn5 (nn.BatchNorm1d): A batch normalization layer applied after fc2.

    Example:
        >>> # Create an instance of the STN3d module
        >>> stn = STN3d()

        >>> # Generate a batch of 3D point clouds
        >>> x = torch.randn(32, 3, 1024)

        >>> # Apply the STN3d module to the input point clouds
        >>> transformed = stn(x)

        >>> # Print the shape of the output transformation matrix
        >>> print(transformed.shape)  # Output: torch.Size([32, 3, 3])
    """

    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(
            np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    """
    Spatial Transformer Network (STN) module with k-dimensional input.

    The STNkd module takes as input a tensor of shape (batch_size, k, n), where
    k is the number of input features and n is the number of input points. It
    performs a series of convolutional and fully-connected layers to learn an
    affine transformation matrix for the input points. The transformation matrix
    is then applied to the input points to obtain the transformed output.

    Args:
        k (int): Number of input features.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, k, k), representing the
        learned affine transformation matrix.

    Examples:
        >>> stn = STNkd(k=3)
        >>> x = torch.randn(2, 3, 100)
        >>> y = stn(x)
        >>> print(y.shape)
        torch.Size([2, 3, 3])
    """

    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(
            np.float32))).view(1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
