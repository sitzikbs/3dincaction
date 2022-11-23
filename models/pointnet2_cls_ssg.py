# Adapted for temporal point clodus from the pytorch pointnet2 repo: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import torch.nn as nn
import torch.nn.functional as F
import torch
from models.pointnet2_utils import PointNetSetAbstraction, PointNetPP4DSetAbstraction


class PointNetPP4D(nn.Module):
    def __init__(self, num_class, n_frames=32, in_channel=3):
        super(PointNetPP4D, self).__init__()
        self.n_frames = n_frames
        self.sa1 = PointNetPP4DSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetPP4DSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetPP4DSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)


    def forward(self, xyz):
        b, t, d, n = xyz.shape
        # new_B = B*t
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.reshape(-1, 1024)

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return {'pred': x.reshape(b, t, -1).permute([0, 2, 1]), 'features': l3_points.reshape(b, t, -1)}

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.fc3 = nn.Linear(256, num_classes)


class PointNet2(nn.Module):
    def __init__(self, num_class, n_frames=32, in_channel=3):
        super(PointNet2, self).__init__()
        self.n_frames = n_frames
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)


    def forward(self, xyz):
        B, t, d, n = xyz.shape
        norm = None
        xyz = xyz.reshape(-1, d, n)
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(-1, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return {'pred': x.reshape(B, t, -1).permute(0, 2, 1), 'features': l3_points.reshape(B, t, -1)}

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.fc3 = nn.Linear(256, num_classes)


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
