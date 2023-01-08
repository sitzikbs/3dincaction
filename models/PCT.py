import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def sample_and_knn_group(s, k, coords, features):
    """
    Sampling by FPS and grouping by KNN.
    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by KNN
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]

    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()

    # FPS sampling
    fps_idx = pointnet2_utils.furthest_point_sample(coords, s).long()  # [B, s]
    new_coords = index_points(coords, fps_idx)  # [B, s, 3]
    new_features = index_points(features, fps_idx)  # [B, s, D]

    # K-nn grouping
    idx = knn_point(k, coords, new_coords)  # [B, s, k]
    grouped_features = index_points(features, idx)  # [B, s, k, D]

    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, k, D]

    # Concat
    aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, k, 1)],
                                    dim=-1)  # [B, s, k, 2D]

    return new_coords, aggregated_features  # [B, s, 3], [B, s, k, 2D]


class Embedding(nn.Module):
    """
    Input Embedding layer which consist of 2 stacked LBR layer.
    """

    def __init__(self, in_channels=3, out_channels=128):
        super(Embedding, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        Input
            x: [B, in_channels, N]

        Output
            x: [B, out_channels, N]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class SA(nn.Module):
    """
    Self Attention module.
    """

    def __init__(self, channels):
        super(SA, self).__init__()

        self.da = channels // 4

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Input
            x: [B, de, N]

        Output
            x: [B, de, N]
        """
        # compute query, key and value matrix
        x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, da]
        x_k = self.k_conv(x)  # [B, da, N]
        x_v = self.v_conv(x)  # [B, de, N]

        # compute attention map and scale, the sorfmax
        energy = torch.bmm(x_q, x_k) / (math.sqrt(self.da))  # [B, N, N]
        attention = self.softmax(energy)  # [B, N, N]

        # weighted sum
        x_s = torch.bmm(x_v, attention)  # [B, de, N]
        x_s = self.act(self.after_norm(self.trans_conv(x_s)))

        # residual
        x = x + x_s

        return x


class SG(nn.Module):
    """
    SG(sampling and grouping) module.
    """

    def __init__(self, s, in_channels, out_channels):
        super(SG, self).__init__()

        self.s = s

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, coords):
        """
        Input:
            x: features with size of [B, in_channels//2, N]
            coords: coordinates data with size of [B, N, 3]
        """
        x = x.permute(0, 2, 1)  # (B, N, in_channels//2)
        new_xyz, new_feature = sample_and_knn_group(s=self.s, k=32, coords=coords,
                                                    features=x)  # [B, s, 3], [B, s, 32, in_channels]
        b, s, k, d = new_feature.size()
        new_feature = new_feature.permute(0, 1, 3, 2)
        new_feature = new_feature.reshape(-1, d, k)  # [Bxs, in_channels, 32]
        batch_size = new_feature.size(0)
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))  # [Bxs, in_channels, 32]
        new_feature = F.relu(self.bn2(self.conv2(new_feature)))  # [Bxs, in_channels, 32]
        new_feature = F.adaptive_max_pool1d(new_feature, 1).view(batch_size, -1)  # [Bxs, in_channels]
        new_feature = new_feature.reshape(b, s, -1).permute(0, 2, 1)  # [B, in_channels, s]
        return new_xyz, new_feature


class NeighborEmbedding(nn.Module):
    def __init__(self, samples=[512, 256]):
        super(NeighborEmbedding, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.sg1 = SG(s=samples[0], in_channels=128, out_channels=128)
        self.sg2 = SG(s=samples[1], in_channels=256, out_channels=256)

    def forward(self, x):
        """
        Input:
            x: [B, 3, N]
        """
        xyz = x.permute(0, 2, 1)  # [B, N ,3]

        features = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        features = F.relu(self.bn2(self.conv2(features)))  # [B, 64, N]

        xyz1, features1 = self.sg1(features, xyz)  # [B, 128, 512]
        _, features2 = self.sg2(features1, xyz1)  # [B, 256, 256]

        return features2


class OA(nn.Module):
    """
    Offset-Attention Module.
    """

    def __init__(self, channels):
        super(OA, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self, x):
        """
        Input:
            x: [B, de, N]

        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here

        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x


class NaivePCT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embedding(3, 128)

        self.sa1 = SA(128)
        self.sa2 = SA(128)
        self.sa3 = SA(128)
        self.sa4 = SA(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.embedding(x)

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class SPCT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embedding(3, 128)

        self.sa1 = OA(128)
        self.sa2 = OA(128)
        self.sa3 = OA(128)
        self.sa4 = OA(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.embedding(x)

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class PCT(nn.Module):
    def __init__(self, samples=[512, 256]):
        super().__init__()

        self.neighbor_embedding = NeighborEmbedding(samples)

        self.oa1 = OA(256)
        self.oa2 = OA(256)
        self.oa3 = OA(256)
        self.oa4 = OA(256)

        self.linear = nn.Sequential(
            nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.neighbor_embedding(x)

        x1 = self.oa1(x)
        x2 = self.oa2(x1)
        x3 = self.oa3(x2)
        x4 = self.oa4(x3)

        x = torch.cat([x, x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class Classification(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_categories)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class Segmentation(nn.Module):
    def __init__(self, part_num):
        super().__init__()

        self.part_num = part_num

        self.label_conv = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)

        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(0.5)

    def forward(self, x, x_max, x_mean, cls_label):
        batch_size, _, N = x.size()

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        x = torch.cat([x, x_max_feature, x_mean_feature, cls_label_feature], dim=1)  # 1024 * 3 + 64

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        return x


class NormalEstimation(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs1 = nn.Conv1d(1024 * 3, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, 3, 1)

        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(0.5)

    def forward(self, x, x_max, x_mean):
        N = x.size(2)

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)

        x = torch.cat([x_max_feature, x_mean_feature, x], dim=1)

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        return x


"""
Classification networks.
"""


class NaivePCTCls(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.encoder = NaivePCT()
        self.cls = Classification(num_categories)

    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x


class SPCTCls(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.encoder = SPCT()
        self.cls = Classification(num_categories)

    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x


class PCTCls(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.encoder = PCT()
        self.cls = Classification(num_categories)

    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x


"""
Part Segmentation Networks.
"""


class NaivePCTSeg(nn.Module):
    def __init__(self, part_num=50):
        super().__init__()

        self.encoder = NaivePCT()
        self.seg = Segmentation(part_num)

    def forward(self, x, cls_label):
        x, x_max, x_mean = self.encoder(x)
        x = self.seg(x, x_max, x_mean, cls_label)
        return x


class SPCTSeg(nn.Module):
    def __init__(self, part_num=50):
        super().__init__()

        self.encoder = SPCT()
        self.seg = Segmentation(part_num)

    def forward(self, x, cls_label):
        x, x_max, x_mean = self.encoder(x)
        x = self.seg(x, x_max, x_mean, cls_label)
        return x


class PCTSeg(nn.Module):
    def __init__(self, part_num=50):
        super().__init__()

        self.encoder = PCT(samples=[1024, 1024])
        self.seg = Segmentation(part_num)

    def forward(self, x, cls_label):
        x, x_max, x_mean = self.encoder(x)
        x = self.seg(x, x_max, x_mean, cls_label)
        return x


"""
Normal Estimation networks.
"""


class NaivePCTNormalEstimation(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = NaivePCT()
        self.ne = NormalEstimation()

    def forward(self, x):
        x, x_max, x_mean = self.encoder(x)
        x = self.ne(x, x_max, x_mean)
        return x


class SPCTNormalEstimation(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = SPCT()
        self.ne = NormalEstimation()

    def forward(self, x):
        x, x_max, x_mean = self.encoder(x)
        x = self.ne(x, x_max, x_mean)
        return x


class PCTNormalEstimation(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = PCT(samples=[1024, 1024])
        self.ne = NormalEstimation()

    def forward(self, x):
        x, x_max, x_mean = self.encoder(x)
        x = self.ne(x, x_max, x_mean)
        return x



class CorrespondanceFinder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.convs1 = nn.Conv1d(1024 * 3, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, out_dim, 1)

        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(0.5)

    def forward(self, x, x_max, x_mean):
        N = x.size(2)

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)

        x = torch.cat([x_max_feature, x_mean_feature, x], dim=1)

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        return x

class PCTCorreformer(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.encoder = PCT(samples=[1024, 1024])
        self.cf = CorrespondanceFinder(out_dim)

    def forward(self, x):
        x, x_max, x_mean = self.encoder(x)
        x = self.cf(x, x_max, x_mean)
        return x


if __name__ == '__main__':
    pc = torch.rand(4, 3, 1024).to('cuda')
    cls_label = torch.rand(4, 16).to('cuda')

    # testing for cls networks
    naive_pct_cls = NaivePCTCls().to('cuda')
    spct_cls = SPCTCls().to('cuda')
    pct_cls = PCTCls().to('cuda')

    print(naive_pct_cls(pc).size())
    print(spct_cls(pc).size())
    print(pct_cls(pc).size())

    # testing for segmentation networks
    naive_pct_seg = NaivePCTSeg().to('cuda')
    spct_seg = SPCTSeg().to('cuda')
    pct_seg = PCTSeg().to('cuda')

    print(naive_pct_seg(pc, cls_label).size())
    print(spct_seg(pc, cls_label).size())
    print(pct_seg(pc, cls_label).size())

    # testing for normal estimation networks
    naive_pct_ne = NaivePCTNormalEstimation().to('cuda')
    spct_ne = SPCTNormalEstimation().to('cuda')
    pct_ne = PCTNormalEstimation().to('cuda')

    print(naive_pct_ne(pc).size())
    print(spct_ne(pc).size())
    print(pct_ne(pc).size())