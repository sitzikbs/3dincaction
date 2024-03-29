import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pstnet_modules import pstnet_pointnet2_utils as pointnet2_utils


def kaiming_uniform(tensor, size):
    fan = size[1] * size[2] * size[3]
    gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def uniform(tensor, a, b):
    with torch.no_grad():
        return tensor.uniform_(a, b)
class PSTConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mid_planes: int,
                 out_planes: int,
                 spatial_kernel_size: [float, int],
                 temporal_kernel_size: int,
                 spatial_stride: int = 1,
                 temporal_stride: int = 1,
                 temporal_padding: [int, int] = [0, 0],
                 padding_mode: str = "zeros",
                 spatial_aggregation: str = "addition",
                 spatial_pooling: str = "max",
                 bias: bool = False,
                 batch_norm: bool = True):
        """
        Args:
            in_planes: C, number of point feature channels in the input. it is 0 if point features are not available.
            mid_planes: C_m, number of channels produced by the spatial convolution
            out_planes: C', number of channels produced by the temporal convolution
            spatial_kernel_size: (r, k), radius and nsamples
            temporal_kernel_size: odd
            spatial_stride: spatial sub-sampling rate, >= 1
            temporal_stride: controls the stride for the temporal cross correlation, >= 1
            temporal_padding:
            padding_mode: "zeros" or "replicate"
            spatial_aggregation: controls the way to aggregate point displacements and point features, "addition" or "multiplication"
            spatial_pooling: "max", "sum" or "avg"
            bias:
            batch_norm:
        """
        super().__init__()

        assert (padding_mode in ["zeros", "replicate"]), "PSTConv: 'padding_mode' should be 'zeros' or 'replicate'!"
        assert (spatial_aggregation in ["addition", "multiplication"]), "PSTConv: 'spatial_aggregation' should be 'addition' or 'multiplication'!"
        assert (spatial_pooling in ["max", "sum", "avg"]), "PSTConv: 'spatial_pooling' should be 'max', 'sum' or 'avg'!"

        self.in_planes = in_planes
        self.mid_planes = mid_planes
        self.out_planes = out_planes

        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride

        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_radius = math.floor(temporal_kernel_size/2)
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding
        self.padding_mode = padding_mode

        self.spatial_aggregation = spatial_aggregation
        self.spatial_pooling = spatial_pooling

        if in_planes != 0:
            self.spatial_conv_f = nn.Conv2d(in_channels=in_planes, out_channels=mid_planes, kernel_size=1, stride=1, padding=0, bias=bias)
            kaiming_uniform(self.spatial_conv_f.weight, size=[mid_planes, in_planes+3, 1, 1])
            if bias:
                bound = 1 / math.sqrt(in_planes+3)
                uniform(self.spatial_conv_f.bias, -bound, bound)

        self.spatial_conv_d = nn.Conv2d(in_channels=3, out_channels=mid_planes, kernel_size=1, stride=1, padding=0, bias=bias)
        kaiming_uniform(self.spatial_conv_d.weight, size=[mid_planes, in_planes+3, 1, 1])
        if bias:
            bound = 1 / math.sqrt(in_planes+3)
            uniform(self.spatial_conv_d.bias, -bound, bound)

        self.batch_norm = nn.BatchNorm1d(num_features=temporal_kernel_size*mid_planes) if batch_norm else False
        self.relu = nn.ReLU(inplace=True)

        self.temporal = nn.Conv1d(in_channels=temporal_kernel_size*mid_planes, out_channels=out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, xyzs: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            xyzs: torch.Tensor
                 (B, L, N, 3) tensor of sequence of the xyz coordinates
            features: torch.Tensor
                 (B, L, C, N) tensor of sequence of the features
        """
        device = xyzs.get_device()

        nframes = xyzs.size(1)  # L
        npoints = xyzs.size(2)  # N

        if self.temporal_kernel_size > 1 and self.temporal_stride > 1:
            assert ((nframes + sum(self.temporal_padding) - self.temporal_kernel_size) % self.temporal_stride == 0), "PSTConv: Temporal parameter error!"

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        if self.in_planes != 0:
            features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
            features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

        if self.padding_mode == "zeros":
            xyz_padding = torch.zeros(xyzs[0].size(), dtype=torch.float32, device=device)
            for i in range(self.temporal_padding[0]):
                xyzs = [xyz_padding] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyz_padding]

            if self.in_planes != 0:
                feature_padding = torch.zeros(features[0].size(), dtype=torch.float32, device=device)
                for i in range(self.temporal_padding[0]):
                    features = [feature_padding] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [feature_padding]
        else:   # "replicate"
            for i in range(self.temporal_padding[0]):
                xyzs = [xyzs[0]] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyzs[-1]]

            if self.in_planes != 0:
                for i in range(self.temporal_padding[0]):
                    features = [features[0]] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [features[-1]]

        new_xyzs = []
        new_features = []
        for t in range(self.temporal_radius, len(xyzs)-self.temporal_radius, self.temporal_stride):                                 # temporal anchor frames
            # spatial anchor point subsampling by FPS
            anchor_idx = pointnet2_utils.furthest_point_sample(xyzs[t], npoints//self.spatial_stride)                               # (B, N//self.spatial_stride)
            anchor_xyz_flipped = pointnet2_utils.gather_operation(xyzs[t].transpose(1, 2).contiguous(), anchor_idx)                 # (B, 3, N//self.spatial_stride)
            anchor_xyz_expanded = torch.unsqueeze(anchor_xyz_flipped, 3)                                                            # (B, 3, N//spatial_stride, 1)
            anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous()                                                            # (B, N//spatial_stride, 3)

            # spatial convolution
            spatial_features = []
            for i in range(t-self.temporal_radius, t+self.temporal_radius+1):
                neighbor_xyz = xyzs[i]

                idx = pointnet2_utils.ball_query(self.r, self.k, neighbor_xyz, anchor_xyz)

                neighbor_xyz_flipped = neighbor_xyz.transpose(1, 2).contiguous()                                                    # (B, 3, N)
                neighbor_xyz_grouped = pointnet2_utils.grouping_operation(neighbor_xyz_flipped, idx)                                # (B, 3, N//spatial_stride, k)

                displacement = neighbor_xyz_grouped - anchor_xyz_expanded                                                           # (B, 3, N//spatial_stride, k)
                displacement = self.spatial_conv_d(displacement)                                                                    # (B, mid_planes, N//spatial_stride, k)

                if self.in_planes != 0:
                    neighbor_feature_grouped = pointnet2_utils.grouping_operation(features[i], idx)                                 # (B, in_planes, N//spatial_stride, k)
                    feature = self.spatial_conv_f(neighbor_feature_grouped)                                                         # (B, mid_planes, N//spatial_stride, k)

                    if self.spatial_aggregation == "addition":
                        spatial_feature = feature + displacement
                    else:
                        spatial_feature = feature * displacement

                else:
                    spatial_feature = displacement

                if self.spatial_pooling == 'max':
                    spatial_feature, _ = torch.max(input=spatial_feature, dim=-1, keepdim=False)                                    # (B, mid_planes, N//spatial_stride)
                elif self.spatial_pooling == 'sum':
                    spatial_feature = torch.sum(input=spatial_feature, dim=-1, keepdim=False)                                       # (B, mid_planes, N//spatial_stride)
                else:
                    spatial_feature = torch.mean(input=spatial_feature, dim=-1, keepdim=False)                                      # (B, mid_planes, N//spatial_stride)

                spatial_features.append(spatial_feature)

            spatial_features = torch.cat(tensors=spatial_features, dim=1, out=None)                                                 # (B, temporal_kernel_size*mid_planes, N//spatial_stride)

            # batch norm and relu
            if self.batch_norm:
                spatial_features = self.batch_norm(spatial_features)

            spatial_features = self.relu(spatial_features)

            # temporal convolution
            spatio_temporal_feature = self.temporal(spatial_features)

            new_xyzs.append(anchor_xyz)
            new_features.append(spatio_temporal_feature)

        new_xyzs = torch.stack(tensors=new_xyzs, dim=1)
        new_features = torch.stack(tensors=new_features, dim=1)

        return new_xyzs, new_features

class MSRAction(nn.Module):
    def __init__(self, model_cfg, num_classes=20):
        super(MSRAction, self).__init__()
        cfg = model_cfg['PSTNET']
        radius = cfg['radius']
        nsamples = cfg['nsamples']
        self.conv1 =  PSTConv(in_planes=0,
                              mid_planes=45,
                              out_planes=64,
                              spatial_kernel_size=[radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv2a = PSTConv(in_planes=64,
                              mid_planes=96,
                              out_planes=128,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[1,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv2b = PSTConv(in_planes=128,
                              mid_planes=192,
                              out_planes=256,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[1,1],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv3a = PSTConv(in_planes=256,
                              mid_planes=284,
                              out_planes=512,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[1,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv3b = PSTConv(in_planes=512,
                              mid_planes=768,
                              out_planes=1024,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[1,1],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv4 =  PSTConv(in_planes=1024,
                              mid_planes=1536,
                              out_planes=2048,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.fc = nn.Linear(2048, num_classes)
    def forward(self, xyzs):

        new_xys, new_features = self.conv1(xyzs, None)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv4(new_xys, new_features)               # (B, L, C, N)

        new_features = torch.mean(input=new_features, dim=-1, keepdim=False)    # (B, L, C)

        new_feature = torch.max(input=new_features, dim=1, keepdim=False)[0]    # (B, C)

        out = self.fc(new_feature)

        return out

class NTU(nn.Module):
    def __init__(self, model_cfg, num_classes=20):
        super(NTU, self).__init__()
        cfg = model_cfg['PSTNET']
        radius = cfg['radius']
        nsamples = cfg['nsamples']
        self.conv1 =  PSTConv(in_planes=0,
                              mid_planes=45,
                              out_planes=64,
                              spatial_kernel_size=[radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0])

        self.conv2a = PSTConv(in_planes=64,
                              mid_planes=96,
                              out_planes=128,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[0,0])

        self.conv2b = PSTConv(in_planes=128,
                              mid_planes=192,
                              out_planes=256,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[0,0])

        self.conv3a = PSTConv(in_planes=256,
                              mid_planes=384,
                              out_planes=512,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[0,0])

        self.conv3b = PSTConv(in_planes=512,
                              mid_planes=768,
                              out_planes=1024,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[0,0])

        self.conv4 =  PSTConv(in_planes=1024,
                              mid_planes=1536,
                              out_planes=2048,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0])

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, xyzs):

        new_xys, new_features = self.conv1(xyzs, None)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv4(new_xys, new_features)               # (B, L, C, N)

        new_features = torch.mean(input=new_features, dim=-1, keepdim=False)    # (B, L, C)

        new_feature = torch.max(input=new_features, dim=1, keepdim=False)[0]    # (B, C)

        out = self.fc(new_feature)

        return out


class PSTnet(MSRAction):
    def __init__(self, model_cfg, num_class=40, n_frames=32):
        MSRAction.__init__(self, model_cfg, num_classes=num_class)
        # self.fc1 = nn.Linear(2048, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(2048, num_class)

    def forward(self, X):
        b, t, d, n = X.shape
        X = X.permute(0, 1, 3, 2)

        new_xys, new_features = self.conv1(X, None)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv4(new_xys, new_features)               # (B, L, C, N)

        new_features = torch.mean(input=new_features, dim=-1, keepdim=False)    # (B, L, C)

        # adapting to per frame prediction
        new_t = int(t/4)
        new_features = new_features.reshape(b * new_t, -1) # t reduces by X4
        # out = self.fc(new_features)
        # out = self.drop1(F.relu(self.bn1(self.fc1(new_features).reshape(b, new_t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512))
        # out = self.drop2(F.relu(self.bn2(self.fc2(out).reshape(b, new_t, 256).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 256))
        out = self.fc3(new_features)

        out = F.interpolate(out.reshape(b, new_t, -1).permute(0, 2, 1), t, mode='linear', align_corners=True)
        out = F.log_softmax(out, 1)
        return {'pred': out}
