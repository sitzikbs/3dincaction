import torch
import torch.nn as nn
import faiss
from faiss.contrib.torch_utils import torch_replacement_knn_gpu as faiss_torch_knn_gpu
import models.pointnet2_utils as utils
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
from scipy.spatial import cKDTree


class PatchletsExtractor(nn.Module):
    def __init__(self, k=16, sample_mode='nn'):
        super(PatchletsExtractor, self).__init__()
        #TODO consider implementing a radius threshold
        self.k = k
        self.sample_mode = sample_mode

        self.res = faiss.StandardGpuResources()
        self.res.setDefaultNullStreamAllDevices()

    def forward(self, point_seq, feat_seq=None):
        b, t, n, d = point_seq.shape
        if feat_seq is None:
            feat_seq = point_seq
            d_feat = d
        else:
            d_feat = feat_seq.shape[-1]

        x1 = point_seq
        x2 = torch.cat([point_seq[:, [0]], point_seq], dim=1)[:, :-1]
        if self.sample_mode == 'nn':
            selected_point_idx = 0

        distances = torch.empty(b*t, n, self.k, device=point_seq.device)
        idxs = torch.empty(b*t, n, self.k, device=point_seq.device, dtype=torch.long)
        patchlets = torch.empty(b*t, n, self.k, device=point_seq.device, dtype=torch.long)

        x1, x2 = x1.reshape(-1, n, d), x2.reshape(-1, n, d)
        feat_seq = feat_seq.reshape(-1, n, d_feat)

        # TODO subsample
        # fps_idx = farthest_point_sample(x1[:, 0], self.npoint)  # [B, npoint, C]
        # x1_subsampled = index_points(x1[:, 0], fps_idx.unsqueeze(1).repeat([1, t, 1]).reshape(-1, npoint))

        distances[0], idxs[0] = faiss_torch_knn_gpu(self.res, x2[0], x1[0], k=self.k)
        patchlets[0] = idxs[0]

        # loop over the data to reorder the indices to form the patchlets
        for i in range(1, len(x1)):
            xb, xq = x1[i], x2[i]
            distances[i], idxs[i] = faiss_torch_knn_gpu(self.res, xq, xb, k=self.k)
            prev_frame_neighbor_idx = patchlets[i - 1, :, selected_point_idx]
            patchlets[i] = idxs[i][prev_frame_neighbor_idx, :]


        patchlet_points = utils.index_points(x1, patchlets)
        patchlet_feats = utils.index_points(feat_seq, patchlets)


        # reshape all to bxtxnxk
        distances, idxs = distances.reshape(b, t, n, self.k), idxs.reshape(b, t, n, self.k)
        patchlets, patchlet_points = patchlets.reshape(b, t, n, self.k), patchlet_points.reshape(b, t, n, self.k, d)
        patchlet_feats = patchlet_feats.reshape(b, t, n, self.k, d_feat)


        return {'idx': idxs, 'distances': distances, 'patchlets': patchlets,
                'patchlet_points': patchlet_points, 'patchlet_feats': patchlet_feats}


class PointNet2PatchletsSA(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, k=16, temporal_conv=4, extract_patchlets=True):
        super(PointNet2PatchletsSA, self).__init__()
        self.k = k
        self.patchlet_extractor = PatchletsExtractor(k=self.k, sample_mode='nn')
        self.extract_patchlets = extract_patchlets

        self.radius = radius
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv3d(last_channel, out_channel, [1, temporal_conv, self.k], 1, padding='same'))
            self.mlp_bns.append(nn.BatchNorm3d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        b, t, k, n = xyz.shape

        # xyz = xyz.reshape(-1, k, n)
        # xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 1, 3, 2)

            # points = points.reshape(-1, points.shape[-2], points.shape[-1])

        if self.patchlet_extractor:
            if points is None:
                patchlet_dict = self.patchlet_extractor(xyz, None)
            else:
                patchlet_dict = self.patchlet_extractor(xyz, points)
            patchlet_features = patchlet_dict['patchlet_feats']
            patchlet_features = patchlet_features.permute(0, 2, 4, 1, 3) # [b, d+k, npoint, t, nsample]
            out_xyz = patchlet_dict['patchlet_points'][:, :, :, 0, :].permute(0, 1, 3, 2)
        else:
            xyz = xyz.reshape(-1, k, n).permute(0, 2, 1)
            if self.group_all:
                new_xyz, new_points = utils.sample_and_group_all_4d(xyz.reshape(b, t, n, k), points)
            else:
                new_xyz, new_points = utils.sample_and_group_4d(self.npoint, self.radius, self.nsample,
                                                                xyz.reshape(b, t, n, k), points)
            patchlet_features = new_points.reshape(b, t, new_points.shape[-3], new_points.shape[-2], new_points.shape[-1])
            out_xyz = new_xyz.reshape(b, t, new_xyz.shape[-2], new_xyz.shape[-1]).permute(0, 1, 3, 2)

            patchlet_features = patchlet_features.permute(0, 4, 2, 1, 3)  # [b, d+k, npoint, t, nsample]

        # new_xyz: sampled points position data, [b*t, npoint, k]
        # new_points: sampled points data, [b*t, npoint, nsample, d+k]

        # patchlet_features = patchlet_features.permute(0, 4, 2, 1, 3)  # [b, d+k, npoint, t, nsample]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            patchlet_features = F.relu(bn(conv(patchlet_features)))

        patchlet_features = torch.max(patchlet_features, -1)[0]
        patchlet_features = patchlet_features.permute(0, 3, 2, 1)

        return out_xyz, patchlet_features.permute(0, 1, 3, 2)




class PointNet2Patchlets(nn.Module):
    def __init__(self, num_class, n_frames=32, in_channel=3):
        super(PointNet2Patchlets, self).__init__()
        self.n_frames = n_frames
        self.patchlet_extractor = PatchletsExtractor(k=self.k, sample_mode='nn')
        self.sa1 = PointNet2PatchletsSA(npoint=512, radius=0.2, nsample=32, in_channel=in_channel,
                                        mlp=[64, 64, 128], group_all=False, k=8, temporal_conv=8,
                                        extract_patchlets=True)
        self.sa2 = PointNet2PatchletsSA(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3,
                                        mlp=[128, 128, 256], group_all=False, k=8, temporal_conv=4,
                                        extract_patchlets=False)
        self.sa3 = PointNet2PatchletsSA(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                        mlp=[256, 512, 1024], group_all=True, k=1, temporal_conv=4,
                                        extract_patchlets=False)

        self.temporal_pool = torch.nn.MaxPool3d([n_frames, 1, 1])
        self.temporal_pool_xyz = torch.nn.AvgPool3d([4, 1, 1])

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

        self.bnt = nn.BatchNorm1d(1024)
        self.temporalconv1 = torch.nn.Conv1d(1024, 1024, int(n_frames/4), 1, padding='same')
        self.temporalconv2 = torch.nn.Conv1d(256, 256, n_frames, 1, padding='same')
        self.bn3 = nn.BatchNorm1d(256)


    def forward(self, xyz):
        b, t, d, n = xyz.shape
        # new_B = B*t
        patchlet_dict = self.patchlet_extractor(xyz)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l3_xyz, l3_points = l3_xyz.squeeze(-1), l3_points.squeeze(-1)
        x = l3_points.permute(0, 2, 1)

        x = F.relu(self.bnt(self.temporalconv1(x)))

        x = x.permute(0, 2, 1).reshape(b*t, 1024)

        x = self.drop1(F.relu(self.bn1(self.fc1(x).reshape(b, t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512))
        x = self.drop2(F.relu(self.bn2(self.fc2(x).reshape(b, t, 256).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 256))
        # learn a temporal filter on all per-frame global representations
        x = F.relu(self.bn3(self.temporalconv2(x.reshape(b, t, 256).permute(0, 2, 1)).permute(0, 2, 1).reshape(-1, 256)))
        x = self.fc3(x)

        x = F.log_softmax(x, -1)

        return {'pred': x.reshape(b, t, -1).permute([0, 2, 1]), 'features': l3_points}