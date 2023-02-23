# Adapted for temporal point clodus from the pytorch pointnet2 repo: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class PointNet2MSG(nn.Module):
    def __init__(self, model_cfg, num_class, n_frames=32, in_channel=3):
        super(PointNet2MSG, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.1, 0.2, 0.4], nsample_list=[16, 32, 128],
                                             in_channel=in_channel, mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=128, radius_list=[0.2, 0.4, 0.8], nsample_list=[32, 64, 128],
                                             in_channel=320+in_channel, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=640 + in_channel,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        b, t, k, n = xyz.shape
        norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.reshape(b * t, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x).reshape(b, t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512))
        x = self.drop2(F.relu(self.bn2(self.fc2(x).reshape(b, t, 256).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 256))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return {'pred': x.reshape(b, t, -1).permute(0, 2, 1), 'features': l3_points.reshape(b, t, -1)}

class PointNet2MSGBasic(nn.Module):
    def __init__(self, model_cfg, num_class, n_frames=32, in_channel=3):
        super(PointNet2MSGBasic, self).__init__()
        self.n_frames = n_frames
        self.sa1 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.1, 0.2, 0.4], nsample_list=[16, 32, 128],
                                             in_channel=in_channel, mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=128, radius_list=[0.2, 0.4, 0.8], nsample_list=[32, 64, 128],
                                             in_channel=320 + in_channel, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=640 + in_channel,
                                          mlp=[256, 512, 1024], group_all=True)

        self.temporalconv = torch.nn.Conv1d(256, 256, n_frames, 1, padding='same')
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        b, t, k, n = xyz.shape
        norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.reshape(b * t, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x).reshape(b, t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512))
        x = self.drop2(F.relu(self.bn2(self.fc2(x).reshape(b, t, 256).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 256))
        # learn a temporal filter on all per-frame global representations
        x = F.relu(self.bn3(self.temporalconv(x.reshape(b, t, 256).permute(0, 2, 1)).permute(0, 2, 1).reshape(-1, 256)))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return {'pred': x.reshape(b, t, -1).permute(0, 2, 1), 'features': l3_points.reshape(b, t, -1)}
