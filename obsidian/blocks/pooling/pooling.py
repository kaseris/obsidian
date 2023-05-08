import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.ops as ops

from obsidian.blocks.pooling.registry_ import POOLING, DESCRIPTORS


@POOLING.register('landmark_pooling')
class LandmarkPooling(nn.Module):

    def __init__(self,
                 pool_plane,
                 inter_channels,
                 outchannels,
                 crop_size=7,
                 img_size=(224, 224),
                 num_lms=8,
                 roi_size=2):
        super(LandmarkPooling, self).__init__()
        self.maxpool = nn.MaxPool2d(pool_plane)
        self.linear = nn.Sequential(
            nn.Linear(num_lms * inter_channels, outchannels), nn.ReLU(True),
            nn.Dropout())

        self.inter_channels = inter_channels
        self.outchannels = outchannels
        self.num_lms = num_lms
        self.crop_size = crop_size
        assert img_size[0] == img_size[
            1], 'img width should equal to img height'
        self.img_size = img_size[0]
        self.roi_size = roi_size

        self.a = self.roi_size / float(self.crop_size)
        self.b = self.roi_size / float(self.crop_size)

    def forward(self, features, landmarks):
        """batch-wise RoI pooling.
        Args:
            features(tensor): the feature maps to be pooled.
            landmarks(tensor): crop the region of interest based on the
                landmarks(bs, self.num_lms).
        """
        batch_size = features.size(0)

        # transfer landmark coordinates from original image to feature map
        landmarks = landmarks / self.img_size * self.crop_size
        landmarks = landmarks.view(batch_size, self.num_lms, 2)

        ab = [np.array([[self.a, 0], [0, self.b]]) for _ in range(batch_size)]
        ab = np.stack(ab, axis=0)
        ab = torch.from_numpy(ab).float().cuda()
        size = torch.Size(
            (batch_size, features.size(1), self.roi_size, self.roi_size))

        pooled = []
        for i in range(self.num_lms):
            tx = -1 + 2 * landmarks[:, i, 0] / float(self.crop_size)
            ty = -1 + 2 * landmarks[:, i, 1] / float(self.crop_size)
            t_xy = torch.stack((tx, ty)).view(batch_size, 2, 1)
            theta = torch.cat((ab, t_xy), 2)

            flowfield = F.affine_grid(theta, size)
            one_pooled = F.grid_sample(
                features,
                flowfield.to(torch.float32),
                mode='bilinear',
                padding_mode='border')
            one_pooled = self.maxpool(one_pooled).view(batch_size,
                                                       self.inter_channels)

            pooled.append(one_pooled)
        pooled = torch.stack(pooled, dim=1).view(batch_size, -1)
        pooled = self.linear(pooled)
        return pooled

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


@POOLING.register('global_pooling')
class GlobalPooling(nn.Module):

    def __init__(self, inplanes, pool_plane, inter_channels, outchannels):
        super(GlobalPooling, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(inplanes)

        inter_plane = inter_channels[0] * inplanes[0] * inplanes[1]
        if len(inter_channels) > 1:
            self.global_layers = nn.Sequential(
                nn.Linear(inter_plane, inter_channels[1]),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(inter_channels[1], outchannels),
                nn.ReLU(True),
                nn.Dropout(),
            )
        else:  # just one linear layer
            self.global_layers = nn.Linear(inter_plane, outchannels)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        global_pool = self.global_layers(x)
        return global_pool

    def init_weights(self):
        if isinstance(self.global_layers, nn.Linear):
            nn.init.normal_(self.global_layers.weight, 0, 0.01)
            if self.global_layers.bias is not None:
                nn.init.constant_(self.global_layers.bias, 0)
        elif isinstance(self.global_layers, nn.Sequential):
            for m in self.global_layers:
                if type(m) == nn.Linear:
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


@POOLING.register('roi_pooling')
class RoIPooling(nn.Module):
    """
    Implements the Region of Interest (RoI) pooling module introduced in
    "Fast R-CNN" (https://arxiv.org/abs/1504.08083).

    Given an input tensor of size [batch_size, feats_dim, w, h], RoIPooling
    produces a fixed-sized output tensor of size [batch_size, n_channels,
    roi_pool_size, roi_pool_size] for each specified RoI. The RoI is defined as
    a rectangle with coordinates (x1, y1, x2, y2), where (x1, y1) is the
    top-left corner, and (x2, y2) is the bottom-right corner. 

    Args:
        feats_dim (int): Channel size of the input feature map.
        n_channels (int): Desired output channel size. Default is 32.
        spatial_dim (int): Spatial dimension of the input feature map. Default is 7.
        roi_pool_size (int): Size of the RoI pooling window. Default is 3.

    Attributes:
        reduce_1x1 (nn.Conv2d): A 1x1 convolutional layer to reduce channel dimension.
        fc_bbox_xy (nn.Linear): A fully-connected layer to predict the center coordinates of the RoI.
        fc_bbox_offset (nn.Linear): A fully-connected layer to predict the offset coordinates of the RoI.
        maxpool (nn.AdaptiveMaxPool2d): An instance of PyTorch's AdaptiveMaxPool2d module.

    Inputs:
        features (torch.Tensor): Input tensor of size [batch_size, feats_dim, w, h].

    Outputs:
        pooled_feats (torch.Tensor): Fixed-sized output tensor of size [batch_size, n_channels, roi_pool_size, roi_pool_size].
        bbox (torch.Tensor): Tensor of predicted bounding boxes of size [batch_size, 5], where each row corresponds to a bounding box 
            (batch_idx, x1, y1, x2, y2) for each RoI in the input.

    Example:
        >>> model = RoIPooling(feats_dim=256, n_channels=64, spatial_dim=14, roi_pool_size=5)
        >>> x = torch.rand((2, 256, 14, 14))
        >>> pooled_feats, bbox = model(x)
    """

    def __init__(self,
                 feats_dim,
                 n_channels=32,
                 spatial_dim=7,
                 roi_pool_size=3) -> None:
        super(RoIPooling, self).__init__()
        self.feats_dim = feats_dim  # channel size from previous layer
        self.n_channels = n_channels  # desired output channel size
        # width and height of the output of the previous layer
        self.spatial_dim = spatial_dim
        self.roi_pool_size = roi_pool_size  # desired output width & height

        self.reduce_1x1 = nn.Conv2d(
            self.feats_dim, self.n_channels, kernel_size=1, stride=1)
        self.fc_bbox_xy = nn.Linear(self.n_channels*self.spatial_dim**2, 2)
        self.fc_bbox_offset = nn.Linear(self.n_channels*self.spatial_dim**2, 2)
        self.maxpool = nn.AdaptiveMaxPool2d(self.roi_pool_size)

    def forward(self, features: torch.Tensor):
        # features has size [batch_size, n_channels, h, w]
        batch_size = features.size(0)

        pre_bbox = self.reduce_1x1(features)
        pre_bbox = pre_bbox.reshape(-1, self.n_channels*self.spatial_dim**2)
        bbox_xy = F.sigmoid(self.fc_bbox_xy(pre_bbox))
        bbox_offset = F.sigmoid(self.fc_bbox_offset(pre_bbox))
        bbox = torch.cat((bbox_xy, bbox_xy + bbox_offset), dim=1)
        # roi_pool method expects a tensor of shape: [K, 5]
        # K: batch_size
        # and a single row should be formatted as:
        # [batch_index, x1, y1, x2, y2]
        batch_indices = torch.arange(
            0, batch_size).view(-1, batch_size).t().float()
        rois = torch.cat((batch_indices, bbox), dim=1)

        pooled_feats = ops.roi_pool(features,
                                    boxes=rois,
                                    output_size=self.roi_pool_size)
        return pooled_feats, bbox

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class GlobalDescriptor(nn.Module):
    def __init__(self, p=1) -> None:
        super(GlobalDescriptor, self).__init__()
        self.p = p

    def forward(self,
                x: torch.Tensor):
        assert x.ndim == 4, 'the input tensor of GlobalDescriptor must be the shape of [B, C, H, W]'
        if self.p == 1:
            return x.mean(dim=[-1, -2])
        elif self.p == float('inf'):
            return torch.flatten(F.adaptive_max_pool2d(x, output_size=(1, 1)), start_dim=1)
        else:
            sum_value = x.pow(self.p).mean(dim=[-1, -2])
            return torch.sign(sum_value) * (torch.abs(sum_value).pow(1.0 / self.p))


class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.dim(
        ) == 2, 'the input tensor of L2Norm must be the shape of [B, C]'
        return F.normalize(x, p=2, dim=-1)


@DESCRIPTORS.register('config_descriptor')
class CombinedGlobalDescriptor(nn.Module):
    """
    Module to compute multiple global descriptors on an input tensor and combine them using a user-defined configuration.

    Args:
        fan_in (int): Number of input features.
        gd_config (str): A string representing the desired combination of global descriptors. The string should consist of one or more of the following characters: 'S', 'M', 'G'. 'S', 'M', and 'G' correspond to spatial average pooling, spatial max pooling, and GeM pooling respectively.
        feat_dim (int): The output feature dimension of each global descriptor. The feature dimension should be divisible by the number of global descriptors in the `gd_config`.

    Raises:
        AssertionError: If `gd_config` is not a valid choice.
        AssertionError: If `feat_dim`  is not evenly divisible by the number of global descriptors (n) specified in gd_config.

    Methods:
        forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: Computes multiple global descriptors on the input tensor and combines them using the configuration specified in `gd_config`. Returns a tensor of concatenated global descriptors and the first global descriptor computed separately.
        init_weights(): Initializes the weights of the linear layers in the network.
    """
    CHOICES = ['S', 'M', 'G', 'SM', 'MS', 'SG',
               'GS', 'MG', 'GM', 'SMG', 'MSG', 'GSM']

    def __init__(self,
                 fan_in: int,
                 gd_config: str,
                 feat_dim: int) -> None:
        super(CombinedGlobalDescriptor, self).__init__()
        assert gd_config.upper(
        ) in CombinedGlobalDescriptor.CHOICES, f'gd_config must be any of the following options: {", ".join(c for c in CombinedGlobalDescriptor.CHOICES)}'

        n = len(gd_config)
        k = feat_dim // n
        assert feat_dim % n == 0, 'the feature dim should be divisible by number of global descriptors'

        self.global_descriptors, self.fc_layers = [], []
        for ch in gd_config:
            if ch.upper() == 'S':
                self.global_descriptors.append(GlobalDescriptor(p=1))
                self.fc_layers.append(nn.Linear(fan_in, k, bias=False))
            elif ch.upper() == 'M':
                self.global_descriptors.append(
                    GlobalDescriptor(p=float('inf')))
                self.fc_layers.append(nn.Linear(fan_in, k, bias=False))
            else:
                self.global_descriptors.append(GlobalDescriptor(p=3))
                self.fc_layers.append(nn.Linear(fan_in, k, bias=False))
        self.global_descriptors = nn.ModuleList(self.global_descriptors)
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def forward(self, x: torch.Tensor):
        global_descriptors = []
        for i in range(len(self.global_descriptors)):
            gd = self.global_descriptors[i](x)
            if i == 0:
                # The first global descriptor is returned separately in case
                # the user desires to use it for classifiation by passing it
                # to another layer.
                first_gd = gd
            gd = self.fc_layers[i](gd)
            global_descriptors.append(gd)
        global_descriptors = F.normalize(
            torch.cat(global_descriptors, dim=-1), dim=-1)
        return global_descriptors, first_gd

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    pass
