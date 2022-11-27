
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 1)
        self.conv3 = torch.nn.Conv2d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(512)  # on FC layers
        self.bn5 = nn.BatchNorm1d(256)  # on FC layers


    def forward(self, x):
        b, t = x.size()[0], x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, -1)[0]

        x = x.permute(0, 2, 1).reshape(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x).reshape(b, t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512)
        x = F.relu(self.bn5(self.fc2(x).reshape(b, t, 256).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 256)
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(b*t,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv2d(k, 64, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 1)
        self.conv3 = torch.nn.Conv2d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        b, k, t, n = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, -1)[0]

        x = x.permute(0, 2, 1).reshape(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x).reshape(b, t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512)
        x = F.relu(self.bn5(self.fc2(x).reshape(b, t, 256).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 256)
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(b*t,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat4D(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, in_d=3, n_frames=32, k_frames=4):
        super(PointNetfeat4D, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv2d(in_d, 64, [7, 1], 1, padding='same')
        self.conv2 = torch.nn.Conv2d(64, 128, [3, 1], 1, padding='same')
        self.conv3 = torch.nn.Conv2d(128, 1024, [3, 1], 1, padding='same')
        self.temporal_pool1 = torch.nn.MaxPool2d([3, 1])
        self.temporal_pool2 = torch.nn.MaxPool2d([2, 1])
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        b, k, t, n = x.size()  # batch, feature_dim , temporal, n_points
        trans = self.stn(x)
        x = torch.bmm(x.permute(0, 2, 3, 1).reshape(b * t, n, k), trans)
        x = x.reshape(b, t, n, k).permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = torch.bmm(x.permute(0, 2, 3, 1).reshape(b * t, n, 64), trans_feat)
            x = x.reshape(b, t, n, 64).permute(0, 3, 1, 2)

        else:
            trans_feat = None

        x = self.temporal_pool1(x)
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.temporal_pool2(x)
        x = self.bn3(self.conv3(x))
        x = torch.max(x, -1)[0]
        x = F.interpolate(x, t, mode='linear', align_corners=True)
        x = x.permute(0, 2, 1).reshape(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            # TODO support temporal segmentation
            x = x.view(-1, 1024, 1).repeat(1, 1, n)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls4D(nn.Module):
    def __init__(self, k=2, feature_transform=False, in_d=3, n_frames=32, k_frames=4):
        super(PointNetCls4D, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat4D(global_feat=True, feature_transform=feature_transform, in_d=in_d,
                                   n_frames=32, k_frames=k_frames)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, k, t, n = x.size()
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x).reshape(b, t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512)
        x = F.relu(self.bn2(self.dropout(self.fc2(x).reshape(b, t, 256).permute(0, 2, 1)))).permute(0, 2,
                                                                                                    1).reshape(-1,
                                                                                                               256)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNet4D(nn.Module):
    def __init__(self, k=2, feature_transform=False, in_d=3, n_frames=32):
        super(PointNet4D, self).__init__()
        self.feature_transform = feature_transform
        self.pn = PointNetCls4D(k=k, feature_transform=feature_transform, in_d=in_d, n_frames=n_frames)

    def forward(self, x):
        b, t, k, n = x.shape
        x, trans, trans_feat = self.pn(x.permute([0, 2, 1, 3]))
        x = x.reshape([b, t, -1]).permute([0, 2, 1])
        return {'pred': x, 'trans': trans, 'trans_feat': trans_feat}

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.pn.fc3 = nn.Linear(256, num_classes)


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False, in_d=3):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, in_d=in_d)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, k, t, n = x.size()
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x).reshape(b, t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512)
        x = F.relu(self.bn2(self.dropout(self.fc2(x).reshape(b, t, 256).permute(0, 2, 1)))).permute(0, 2, 1).reshape(-1, 256)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False, in_d=3):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv2d(in_d, 64, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 1)
        self.conv3 = torch.nn.Conv2d(128, 1024, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        b, k, t, n = x.size() # batch, feature_dim , temporal, n_points
        trans = self.stn(x)
        x = torch.bmm(x.permute(0, 2, 3, 1).reshape(b*t, n, k), trans)
        x = x.reshape(b, t, n, k).permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = torch.bmm(x.permute(0, 2, 3, 1).reshape(b*t, n, 64), trans_feat)
            x = x.reshape(b, t, n, 64).permute(0, 3, 1, 2)

        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, -1)[0]
        x = x.permute(0, 2, 1).reshape(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            #TODO support temporal segmentation
            x = x.view(-1, 1024, 1).repeat(1, 1, n)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNet1(nn.Module):
    def __init__(self, k=2, feature_transform=False, in_d=3):
        super(PointNet1, self).__init__()
        self.feature_transform = feature_transform
        self.pn = PointNetCls(k=k, feature_transform=feature_transform, in_d=in_d)

    def forward(self, x):
        b, t, k, n = x.shape
        x, trans, trans_feat = self.pn(x.permute([0, 2, 1, 3]))
        x = x.reshape([b, t, -1]).permute([0, 2, 1])
        return {'pred': x, 'trans': trans, 'trans_feat': trans_feat}




class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
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
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())