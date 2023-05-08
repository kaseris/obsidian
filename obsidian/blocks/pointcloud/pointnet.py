import torch
import torch.nn as nn
import torch.nn.functional as F


from pytorch_metric_learning import samplers, miners, losses, distances, reducers
import obsidian.module as om
from .stn import STN3d, STNkd


class PointNetfeat(om.OBSModule):
    def __init__(self, global_feat=True, feature_transform=False,
                 miner: miners.BaseMiner = None,
                 reducer: reducers.BaseReducer = None,
                 distance: distances.BaseDistance = None,
                 triplet_loss: losses.BaseMetricLossFunction = None,
                 feature_transform_lambda: float = 0.001
                 ):
        super(PointNetfeat, self).__init__()

        self.miner = miner
        self.reducer = reducer
        self.distance = distance
        self.triplet_loss = triplet_loss

        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

    def training_step(self, x: torch.Tensor,
                      targets: torch.Tensor,
                      device,
                      optimizer: torch.optim.Optimizer,
                      scaler: torch.cuda.amp.grad_scaler.GradScaler,
                      lr_scheduler: torch.optim.lr_scheduler.LRScheduler):
        self.train()
        x = x.to(device)
        targets = targets[:, 0]
        targets = targets.to(device)
        x = x.transpose(2, 1)

        with torch.cuda.amp.autocast(enabled=True):
            predictions, trans, trans_feat, global_feat = self(x)
        pairs = self.miner(global_feat, targets)
        loss_cls = F.nll_loss(predictions, targets)
        tr_loss = self.triplet_loss(global_feat, targets, pairs)
        if self.feature_transform:
            feature_tf_loss = feature_transform_regularizer(
                trans_feat) * self.feature_transform_lambda
            loss_cls += feature_tf_loss

        net_loss = loss_cls + tr_loss
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(net_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            net_loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Calculate the accuracy of predictions
        _, predictions = predictions.max(1)
        accuracy = (predictions == targets).sum().item() / targets.size(0)

        return_dict = {'total loss': net_loss.item(),
                       'loss_cls': loss_cls.item(),
                       'loss_tr': tr_loss.item(),
                       'accuracy': accuracy}

        return return_dict

    @torch.no_grad()
    def validation_step(self,
                        x: torch.Tensor,
                        targets: torch.Tensor,
                        device):
        self.eval()
        x = x.to(device=device)
        targets = targets.to(device=device)
        targets = targets[:, 0]
        x = x.transpose(2, 1)

        preds, _, _, _ = self(x)
        loss_cls = F.nll_loss(preds, targets)

        # Calculate the accuracy of predictions
        _, predictions = preds.max(1)
        accuracy = (predictions == targets).sum().item() / targets.size(0)
        return_dict = {'total loss': loss_cls.item(),
                       'accuracy': accuracy}
        return return_dict

    @property
    def params(self):
        return filter(lambda p: p.requires_grad, self.parameters())


class PointNetCls(om.OBSModule):
    def __init__(self, k=2, feature_transform=False,
                 miner: miners.BaseMiner = None,
                 reducer: reducers.BaseReducer = None,
                 distance: distances.BaseDistance = None,
                 triplet_loss: losses.BaseMetricLossFunction = None,
                 feature_transform_lambda: float = 0.001
                 ):
        super(PointNetCls, self).__init__()

        self.miner = miner
        self.reducer = reducer
        self.distance = distance
        self.triplet_loss = triplet_loss

        self.feature_transform_lambda = feature_transform_lambda

        self.feature_transform = feature_transform
        self.feat = PointNetfeat(
            global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        glob_feature = x.clone()
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat, glob_feature

    def training_step(self, x: torch.Tensor,
                      targets: torch.Tensor,
                      device,
                      optimizer: torch.optim.Optimizer,
                      scaler: torch.cuda.amp.grad_scaler.GradScaler,
                      lr_scheduler: torch.optim.lr_scheduler.LRScheduler):
        self.train()
        x = x.to(device)
        targets = targets[:, 0]
        targets = targets.to(device)
        x = x.transpose(2, 1)

        with torch.cuda.amp.autocast(enabled=True):
            predictions, trans, trans_feat, global_feat = self(x)
        pairs = self.miner(global_feat, targets)
        loss_cls = F.nll_loss(predictions, targets)
        self.triplet_loss = self.triplet_loss.to(device)
        tr_loss = self.triplet_loss(global_feat, targets, pairs)
        if self.feature_transform:
            feature_tf_loss = feature_transform_regularizer(
                trans_feat) * self.feature_transform_lambda
            loss_cls += feature_tf_loss

        net_loss = loss_cls + tr_loss
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(net_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            net_loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Calculate the accuracy of predictions
        _, predictions = predictions.max(1)
        accuracy = (predictions == targets).sum().item() / targets.size(0)

        return_dict = {'total loss': net_loss.item(),
                       'loss_cls': loss_cls.item(),
                       'loss_tr': tr_loss.item(),
                       'accuracy': accuracy}

        return return_dict

    @torch.no_grad()
    def validation_step(self,
                        x: torch.Tensor,
                        targets: torch.Tensor,
                        device):
        self.eval()
        x = x.to(device=device)
        targets = targets.to(device=device)
        targets = targets[:, 0]
        x = x.transpose(2, 1)

        preds, _, _, _ = self(x)
        loss_cls = F.nll_loss(preds, targets)

        # Calculate the accuracy of predictions
        _, predictions = preds.max(1)
        accuracy = (predictions == targets).sum().item() / targets.size(0)
        return_dict = {'total loss': loss_cls.item(),
                       'accuracy': accuracy}
        return return_dict

    @property
    def params(self):
        return filter(lambda p: p.requires_grad, self.parameters())


class PointNetDenseCls(om.OBSModule):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(
            global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(
        torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
