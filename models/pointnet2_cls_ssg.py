# Adapted for temporal point clodus from the pytorch pointnet2 repo: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import torch.nn as nn
import torch.nn.functional as F
import torch
from models.pointnet2_utils import PointNetSetAbstraction, PointletSetAbstraction


class PointNetPP4D(nn.Module):
    def __init__(self, num_class, n_frames=32, in_channel=3):
        super(PointNetPP4D, self).__init__()
        self.n_frames = n_frames
        self.sa1 = PointletSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointletSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointletSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        # self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=256 + 3, mlp=[128, 128, 256],
        #                                   group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

        self.maxpool1 = torch.nn.MaxPool3d(kernel_size=[3, 1, 1], stride=[2, 1, 1])
        self.pointavgpool = torch.nn.AvgPool3d(kernel_size=[3, 1, 1], stride=[2, 1, 1])
        # self.maxpool3 = torch.nn.MaxPool3d(kernel_size=[9, 1, 1], stride=[6, 1, 1])

    def forward(self, xyz):
        B, t, d, n = xyz.shape
        norm = None
        # xyz = xyz.reshape(-1, d, n)
        # new_B = B*t
        l1_xyz, l1_points = self.sa1(xyz, norm)

        # temporal_x = l1_points.reshape([-1, t, 128, 512])
        l1_points = self.maxpool1(l1_points)

        # temporal_x = F.interpolate(temporal_x.permute(0, 2, 1, 3), size=[32, 512], mode='bilinear').permute(0, 2, 1, 3).reshape(-1, 128, 512)
        # l1_points = torch.cat([l1_points, temporal_x], dim=1)
        # l1_points = temporal_x
        # t2 = l1_points.shape[1]
        l1_xyz = self.pointavgpool(l1_xyz)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        # temporal_x = l2_points.reshape([-1, t2, 256, 128])
        l2_points = self.maxpool1(l2_points)

        l2_xyz = self.pointavgpool(l2_xyz)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        t3 = l3_points.shape[1]
        x = l3_points.view(-1, 1024)

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        x = F.interpolate(x.reshape(B, t3, -1).permute(0, 2, 1), t, mode='linear')
        return {'pred': x, 'features': l3_points.reshape(B, t, -1)}

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.fc3 = nn.Linear(256, num_classes)

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
