import torch
import torch.nn as nn
import faiss
from faiss.contrib.torch_utils import torch_replacement_knn_gpu as faiss_torch_knn_gpu
import models.pointnet2_utils as utils
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
from scipy.spatial import cKDTree
import numpy as np
from pykeops.torch import Vi, Vj
import time
import models.set_transformer as set_transformer
# import torch_cluster

def get_knn(x1, x2, k=16, res=None, method='faiss_gpu', radius=None):

    if method == 'faiss_gpu':
        distances, idxs = faiss_torch_knn_gpu(res, x1, x2, k=k)
    if method == 'faiss_cpu':
        distances, idxs = faiss.knn(np.ascontiguousarray(x1.detach().cpu().numpy(), dtype=np.float32),
                                    np.ascontiguousarray(x2.detach().cpu().numpy(), dtype=np.float32), k=k)
        distances, idxs = torch.tensor(distances, device=x1.device).cuda(), torch.tensor(idxs, device=x1.device).cuda()
    if method == 'spatial':
        tree = cKDTree(x2.detach().cpu().numpy())
        distances, idxs = tree.query(x1.detach().cpu().numpy(), k, workers=-1)
        distances, idxs = torch.tensor(distances).cuda(), torch.tensor(idxs).cuda()
    if method == 'keops': #supports batch operaations
        X_i = Vi(0, x1.shape[-1])
        X_j = Vj(1, x2.shape[-1])
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        KNN_fun = D_ij.Kmin_argKmin(k, dim=1)
        distances, idxs = KNN_fun(x1.contiguous(), x2.contiguous())

    if radius is not None:
        # clip samples outside a radius, replace with origin to keep k constant
        clipped_idxs = idxs[..., [0]].repeat(1, 1, k)
        mask = distances > radius**2
        idxs[mask] = clipped_idxs[mask]
        distances[mask] = 0
    return distances, idxs


class PatchletsExtractor(nn.Module):
    def __init__(self, k=16, sample_mode='nn', npoints=None, add_centroid_jitter=None, downsample_method=None,
                 radius=None):
        super(PatchletsExtractor, self).__init__()
        self.k = k
        self.radius = radius
        self.sample_mode = sample_mode
        self.downsample_method = downsample_method
        self.npoints = npoints
        self.add_centroid_jitter = add_centroid_jitter
        self.res = faiss.StandardGpuResources()
        self.res.setDefaultNullStreamAllDevices()


    def forward(self, point_seq, feat_seq=None):
        b, t, n, d = point_seq.shape
        n_original = n
        n_out = n
        if feat_seq is None:
            feat_seq = point_seq
            d_feat = d
        else:
            d_feat = feat_seq.shape[-1]

        x1 = point_seq
        x2 = torch.cat([point_seq[:, [0]], point_seq], dim=1)[:, :-1]

        out_x = torch.empty(b, t, n_out, d)
        patchlets = torch.empty(b, t, n, self.k, device=point_seq.device, dtype=torch.long)
        distances_i = torch.empty(b,  t, n, self.k, device=point_seq.device)
        idxs_i = torch.empty(b, t, n, self.k, device=point_seq.device, dtype=torch.long)
        patchlet_points = torch.empty(b, t, n, self.k, d, device=point_seq.device)
        patchlet_feats = torch.empty(b, t, n, self.k, d_feat, device=point_seq.device)

        # loop over the data to reorder the indices to form the patchlets
        x_current = x2[:, 0]
        feat_seq_2 = torch.cat([feat_seq[:, [0]], feat_seq], dim=1)[:, :-1]
        for i in range(0, t):
            x_next = x1[:, i]
            distances, idxs = get_knn(x_current[..., 0:3], x_next[..., 0:3], k=self.k, res=self.res, method='keops', radius=self.radius)
            if self.sample_mode == 'nn':
                x_current = utils.index_points(x_next, idxs)[:, :, 0, :]
            elif self.sample_mode == 'randn':
                rand_idx = torch.randint(self.k, (b, n, 1), device=x_next.device, dtype=torch.int64).repeat([1, 1, 3]).unsqueeze(2)
                x_current = torch.gather(utils.index_points(x_next, idxs).squeeze(), dim=2, index=rand_idx).squeeze()
            elif self.sample_mode == 'gt':
                # only works when point correspondence is known and points are already aligned
                # if self.downsample_method == 'fps':
                #     x_current = utils.index_points(x_next, fps_idx).contiguous()
                # else:
                x_current = x_next
            else:
                raise ValueError("sample mode not supported")

            out_x[:, i] = x_current
            if self.add_centroid_jitter is not None  and not self.sample_mode == 'gt':
                x_current = x_current + self.add_centroid_jitter*torch.randn_like(x_current)

            distances_i[:, i], idxs_i[:, i] = distances, idxs
            patchlets[:, i] = idxs_i[:, i]
            patchlet_points[:, i] = utils.index_points(x_next, idxs).squeeze()
            patchlet_feats[:, i] = utils.index_points(feat_seq_2[:, i], idxs).squeeze()

        distances = distances_i
        idxs = idxs_i

        patchlet_feats = patchlet_feats.reshape(b*t, n, self.k, d_feat)
        patchlet_points = patchlet_points.reshape(b * t, n, self.k, d)
        idxs = idxs.reshape(b*t, n, self.k)
        distances = distances.reshape(b*t, n, self.k)
        patchlets = patchlets.reshape(b*t, n, self.k)

        fps_idx = None
        if self.downsample_method == 'fps':
            #select a subset of the points using fps for maximum coverage
            selected_idxs = utils.farthest_point_sample(point_seq[:, 0].contiguous(), self.npoints).to(torch.int64)
            selected_idxs = selected_idxs.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints)
            patchlet_points = utils.index_points(patchlet_points, selected_idxs)
            patchlet_feats = utils.index_points(patchlet_feats, selected_idxs)
            distances = utils.index_points(distances, selected_idxs)
            idxs = utils.index_points(idxs, selected_idxs)
            patchlets = utils.index_points(patchlets, selected_idxs)
            n = self.npoints
        elif self.downsample_method == 'var' or self.downsample_method == 'mean_var_t':
            # select a subset of the points with the largest point variance for maximum temporal movement
            if self.downsample_method == 'var':
                temporal_patchlet_points = patchlet_points.reshape(b, t, n, self.k, d).permute(0, 2, 1, 3, 4).reshape(b,n,-1,d)
                patchlet_variance = torch.linalg.norm(torch.var(temporal_patchlet_points, -2), dim=-1)
            elif self.downsample_method == 'mean_var_t':
                patchlet_variance = torch.linalg.norm(torch.var(torch.mean(patchlet_points.reshape(b, t, n, self.k, d), -2), 1), dim=-1)
            else:
                raise ValueError("downsample method not supported ")
            _, selected_idxs = torch.topk(patchlet_variance, self.npoints)
            selected_idxs = selected_idxs.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints)
            patchlet_points = utils.index_points(patchlet_points, selected_idxs)
            patchlet_feats = utils.index_points(patchlet_feats, selected_idxs)
            distances = utils.index_points(distances, selected_idxs)
            idxs = utils.index_points(idxs, selected_idxs)
            patchlets = utils.index_points(patchlets, selected_idxs)
            n = self.npoints


        # reshape all to bxtxnxk
        distances, idxs = distances.reshape(b, t, n, self.k), idxs.reshape(b, t, n, self.k)
        patchlets, patchlet_points = patchlets.reshape(b, t, n, self.k), patchlet_points.reshape(b, t, n, self.k, d)
        patchlet_feats = patchlet_feats.reshape(b, t, n, self.k, d_feat)

        normalized_patchlet_points = patchlet_points - patchlet_points[:, 0, :, [0], :].unsqueeze(1).detach() # normalize the patchlet around the center point of the first frame
        patchlet_feats = torch.cat([patchlet_feats, normalized_patchlet_points], -1)

        return {'idx': idxs, 'distances': distances, 'patchlets': patchlets,
                'patchlet_points': patchlet_points, 'patchlet_feats': patchlet_feats,
                'normalized_patchlet_points': normalized_patchlet_points, 'fps_idx': fps_idx,
                'x_current': out_x.reshape(b, t, n_out, d)}

class PatchletsExtractorStrided(nn.Module):
    def __init__(self, k=16, sample_mode='nn', npoints=None, add_centroid_jitter=None, downsample_method=None,
                 radius=None, temporal_stride=8):
        super(PatchletsExtractorStrided, self).__init__()
        #TODO consider implementing a radius threshold
        self.k = k
        self.radius = radius
        self.sample_mode = sample_mode
        self.downsample_method = downsample_method
        self.npoints = npoints
        self.add_centroid_jitter = add_centroid_jitter
        self.res = faiss.StandardGpuResources()
        self.res.setDefaultNullStreamAllDevices()
        self.temporal_stride = temporal_stride

    def get_tpatches(self, x1, x2, feat_seq, flip=False):

        b, t, n, d = x1.shape
        n_out = n

        if feat_seq is None:
            feat_seq = x1
            d_feat = d
        else:
            d_feat = feat_seq.shape[-1]

        out_x = torch.empty(b, t, n_out, d)
        patchlets = torch.empty(b, t, n, self.k, device=x1.device, dtype=torch.long)
        distances_i = torch.empty(b, t, n, self.k, device=x1.device)
        idxs_i = torch.empty(b, t, n, self.k, device=x1.device, dtype=torch.long)
        patchlet_points = torch.empty(b, t, n, self.k, d, device=x1.device)
        patchlet_feats = torch.empty(b, t, n, self.k, d_feat, device=x1.device)

        # loop over the data to reorder the indices to form the patchlets
        x_current = x2[:, 0]
        feat_seq_2 = torch.cat([feat_seq[:, [0]], feat_seq], dim=1)[:, :-1]
        for i in range(0, t):
            x_next = x1[:, i]
            distances, idxs = get_knn(x_current[..., 0:3], x_next[..., 0:3], k=self.k, res=self.res, method='keops',
                                      radius=self.radius)
            if self.sample_mode == 'nn':
                x_current = utils.index_points(x_next, idxs)[:, :, 0, :]
            elif self.sample_mode == 'randn':
                rand_idx = torch.randint(self.k, (b, n, 1), device=x_next.device, dtype=torch.int64).repeat(
                    [1, 1, 3]).unsqueeze(2)
                x_current = torch.gather(utils.index_points(x_next, idxs).squeeze(), dim=2, index=rand_idx).squeeze()
            elif self.sample_mode == 'gt':
                # only works when point correspondence is known and points are already aligned
                # if self.downsample_method == 'fps':
                #     x_current = utils.index_points(x_next, fps_idx).contiguous()
                # else:
                x_current = x_next
            else:
                raise ValueError("sample mode not supported")

            out_x[:, i] = x_current
            if self.add_centroid_jitter is not None  and not self.sample_mode == 'gt':
                x_current = x_current + self.add_centroid_jitter * torch.randn_like(x_current)

            distances_i[:, i], idxs_i[:, i] = distances, idxs
            patchlets[:, i] = idxs_i[:, i]
            patchlet_points[:, i] = utils.index_points(x_next, idxs).squeeze()
            patchlet_feats[:, i] = utils.index_points(feat_seq_2[:, i], idxs).squeeze()

        distances = distances_i
        idxs = idxs_i
        if flip: # reverse temporal order
            patchlet_feats = torch.flip(patchlet_feats, [1])
            patchlet_points = torch.flip(patchlet_points, [1])
            idxs = torch.flip(idxs, [1])
            distances = torch.flip(distances, [1])
            patchlets = torch.flip(patchlets, [1])

        patchlet_feats = patchlet_feats.reshape(b * t, n, self.k, d_feat)
        patchlet_points = patchlet_points.reshape(b * t, n, self.k, d)
        idxs = idxs.reshape(b * t, n, self.k)
        distances = distances.reshape(b * t, n, self.k)
        patchlets = patchlets.reshape(b * t, n, self.k)

        fps_idx = None
        if self.downsample_method == 'fps':
            # select a subset of the points using fps for maximum coverage
            selected_idxs = utils.farthest_point_sample(x1[:, 0].contiguous(), self.npoints).to(torch.int64)
            selected_idxs = selected_idxs.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints)
            patchlet_points = utils.index_points(patchlet_points, selected_idxs)
            patchlet_feats = utils.index_points(patchlet_feats, selected_idxs)
            distances = utils.index_points(distances, selected_idxs)
            idxs = utils.index_points(idxs, selected_idxs)
            patchlets = utils.index_points(patchlets, selected_idxs)
            n = self.npoints
        elif self.downsample_method == 'var' or self.downsample_method == 'mean_var_t':
            # select a subset of the points with the largest point variance for maximum temporal movement
            if self.downsample_method == 'var':
                temporal_patchlet_points = patchlet_points.reshape(b, t, n, self.k, d).\
                    permute(0, 2, 1, 3, 4).reshape(b, n,-1, d)
                patchlet_variance = torch.linalg.norm(torch.var(temporal_patchlet_points, -2), dim=-1)
            elif self.downsample_method == 'mean_var_t':
                patchlet_variance = torch.linalg.norm(
                    torch.var(torch.mean(patchlet_points.reshape(b, t, n, self.k, d), -2), 1), dim=-1)
            else:
                raise ValueError("downsample method not supported ")
            _, selected_idxs = torch.topk(patchlet_variance, self.npoints)
            selected_idxs = selected_idxs.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints)
            patchlet_points = utils.index_points(patchlet_points, selected_idxs)
            patchlet_feats = utils.index_points(patchlet_feats, selected_idxs)
            distances = utils.index_points(distances, selected_idxs)
            idxs = utils.index_points(idxs, selected_idxs)
            patchlets = utils.index_points(patchlets, selected_idxs)
            n = self.npoints

        return patchlet_points, patchlet_feats, distances, idxs, patchlets, n, d_feat, fps_idx, out_x

    def forward(self, point_seq, feat_seq=None):
        b, t, n, d = point_seq.shape
        n_original = n
        n_out = n

        assert t % self.temporal_stride == 0
        n_temporal_segments = int(t / self.temporal_stride)
        patchlets_accum, patchlet_points_accum, patchlet_feats_accum, distances_accum,\
            idxs_accum, out_x_accum = [], [], [], [], [], []
        for i in range(n_temporal_segments):
            x1 = point_seq[:, i*self.temporal_stride:(i+1)*self.temporal_stride]
            x2 = torch.cat([x1[:, [0]], x1], dim=1)[:, :-1]
            feats = feat_seq[:, i*self.temporal_stride:(i+1)*self.temporal_stride] if feat_seq is not None else None
            patchlet_points, patchlet_feats, distances, idxs, patchlets, n, d_feat, fps_idx, out_x = \
                self.get_tpatches(x1, x2, feats, flip=False)

            #bidirectional
            x1 = torch.flip(x1, [1])
            x2 = torch.cat([x1[:, [0]], x1], dim=1)[:, :-1]
            patchlet_points2, patchlet_feats2, distances2, idxs2, patchlets2, _, _, fps_idx2, out_x2 = \
                self.get_tpatches(x1, x2, feat_seq, flip=True)

            # randomly select a subset
            rand_idxs = torch.randperm(n)[:int(n / 2)]
            patchlets = torch.cat([patchlets[:, rand_idxs, :], patchlets2[:, rand_idxs, :]], 1)
            patchlet_feats = torch.cat([patchlet_feats[:, rand_idxs, :], patchlet_feats2[:, rand_idxs, :]], 1)
            patchlet_points = torch.cat([patchlet_points[:, rand_idxs, :], patchlet_points2[:, rand_idxs, :]], 1)
            distances = torch.cat([distances[:, rand_idxs, :], distances2[:, rand_idxs, :]], 1)
            idxs = torch.cat([idxs[:, rand_idxs, :], idxs2[:, rand_idxs, :]], 1)
            out_x = out_x  # remove after debug, out_x is unused

            patchlets_accum.append(patchlets.reshape(b, self.temporal_stride, n, self.k))
            patchlet_points_accum.append(patchlet_points.reshape(b, self.temporal_stride, n, self.k, d))
            patchlet_feats_accum.append(patchlet_feats.reshape(b, self.temporal_stride, n, self.k, d_feat))
            distances_accum.append(distances.reshape(b, self.temporal_stride, n, self.k))
            idxs_accum.append(idxs.reshape(b, self.temporal_stride, n, self.k))
            out_x_accum.append(out_x.reshape(b, self.temporal_stride, n_original, d))

        patchlets = torch.cat(patchlets_accum, 1)
        patchlet_points = torch.cat(patchlet_points_accum, 1)
        patchlet_feats = torch.cat(patchlet_feats_accum, 1)
        distances = torch.cat(distances_accum, 1)
        idxs = torch.cat(idxs_accum, 1)
        out_x = torch.cat(out_x_accum, 1)

        strided_origin_indices = torch.arange(start=0, end=t, step=self.temporal_stride, dtype=torch.int64, device=patchlet_points.device)
        strided_origins = patchlet_points[:, :, :, [0], :].index_select(1, strided_origin_indices)
        trajectetories = patchlet_points[:, :, :, [0], :]
        normalized_patchlet_points = patchlet_points - trajectetories # normalize the patchlet around the center point of the first frame
        trajectetories = trajectetories - torch.repeat_interleave(strided_origins, self.temporal_stride, dim=1)
        patchlet_feats = torch.cat([patchlet_feats, normalized_patchlet_points], -1)

        return {'idx': idxs, 'distances': distances, 'patchlets': patchlets,
                'patchlet_points': patchlet_points, 'patchlet_feats': patchlet_feats,
                'normalized_patchlet_points': normalized_patchlet_points, 'fps_idx': fps_idx,
                'x_current': out_x.reshape(b, t, n_out, d),
                'trajectories': trajectetories.squeeze(3)}


class PatchletsExtractorBidirectional(nn.Module):
    def __init__(self, k=16, sample_mode='nn', npoints=None, add_centroid_jitter=None, downsample_method=None,
                 radius=None):
        super(PatchletsExtractorBidirectional, self).__init__()
        #TODO consider implementing a radius threshold
        self.k = k
        self.radius = radius
        self.sample_mode = sample_mode
        self.downsample_method = downsample_method
        self.npoints = npoints
        self.add_centroid_jitter = add_centroid_jitter
        self.res = faiss.StandardGpuResources()
        self.res.setDefaultNullStreamAllDevices()

    def get_tpatches(self, x1, x2, feat_seq, flip=False):

        b, t, n, d = x1.shape
        n_out = n

        if feat_seq is None:
            feat_seq = x1
            d_feat = d
        else:
            d_feat = feat_seq.shape[-1]

        out_x = torch.empty(b, t, n_out, d)
        patchlets = torch.empty(b, t, n, self.k, device=x1.device, dtype=torch.long)
        distances_i = torch.empty(b, t, n, self.k, device=x1.device)
        idxs_i = torch.empty(b, t, n, self.k, device=x1.device, dtype=torch.long)
        patchlet_points = torch.empty(b, t, n, self.k, d, device=x1.device)
        patchlet_feats = torch.empty(b, t, n, self.k, d_feat, device=x1.device)

        # loop over the data to reorder the indices to form the patchlets
        x_current = x2[:, 0]
        feat_seq_2 = torch.cat([feat_seq[:, [0]], feat_seq], dim=1)[:, :-1]
        for i in range(0, t):
            x_next = x1[:, i]
            distances, idxs = get_knn(x_current[..., 0:3], x_next[..., 0:3], k=self.k, res=self.res, method='keops',
                                      radius=self.radius)
            if self.sample_mode == 'nn':
                x_current = utils.index_points(x_next, idxs)[:, :, 0, :]
            elif self.sample_mode == 'randn':
                rand_idx = torch.randint(self.k, (b, n, 1), device=x_next.device, dtype=torch.int64).repeat(
                    [1, 1, 3]).unsqueeze(2)
                x_current = torch.gather(utils.index_points(x_next, idxs).squeeze(), dim=2, index=rand_idx).squeeze()
            elif self.sample_mode == 'gt':
                # only works when point correspondence is known and points are already aligned
                # if self.downsample_method == 'fps':
                #     x_current = utils.index_points(x_next, fps_idx).contiguous()
                # else:
                x_current = x_next
            else:
                raise ValueError("sample mode not supported")

            out_x[:, i] = x_current
            if self.add_centroid_jitter is not None and not self.sample_mode == 'gt':
                x_current = x_current + self.add_centroid_jitter * torch.randn_like(x_current)

            distances_i[:, i], idxs_i[:, i] = distances, idxs
            patchlets[:, i] = idxs_i[:, i]
            patchlet_points[:, i] = utils.index_points(x_next, idxs).squeeze()
            patchlet_feats[:, i] = utils.index_points(feat_seq_2[:, i], idxs).squeeze()

        distances = distances_i
        idxs = idxs_i
        if flip: # reverse temporal order
            patchlet_feats = torch.flip(patchlet_feats, [1])
            patchlet_points = torch.flip(patchlet_points, [1])
            idxs = torch.flip(idxs, [1])
            distances = torch.flip(distances, [1])
            patchlets = torch.flip(patchlets, [1])

        patchlet_feats = patchlet_feats.reshape(b * t, n, self.k, d_feat)
        patchlet_points = patchlet_points.reshape(b * t, n, self.k, d)
        idxs = idxs.reshape(b * t, n, self.k)
        distances = distances.reshape(b * t, n, self.k)
        patchlets = patchlets.reshape(b * t, n, self.k)

        fps_idx = None
        if self.downsample_method == 'fps':
            # select a subset of the points using fps for maximum coverage
            selected_idxs = utils.farthest_point_sample(x1[:, 0].contiguous(), self.npoints).to(torch.int64)
            selected_idxs = selected_idxs.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints)
            patchlet_points = utils.index_points(patchlet_points, selected_idxs)
            patchlet_feats = utils.index_points(patchlet_feats, selected_idxs)
            distances = utils.index_points(distances, selected_idxs)
            idxs = utils.index_points(idxs, selected_idxs)
            patchlets = utils.index_points(patchlets, selected_idxs)
            n = self.npoints
        elif self.downsample_method == 'var' or self.downsample_method == 'mean_var_t':
            # select a subset of the points with the largest point variance for maximum temporal movement
            if self.downsample_method == 'var':
                temporal_patchlet_points = patchlet_points.reshape(b, t, n, self.k, d).\
                    permute(0, 2, 1, 3, 4).reshape(b, n,-1, d)
                patchlet_variance = torch.linalg.norm(torch.var(temporal_patchlet_points, -2), dim=-1)
            elif self.downsample_method == 'mean_var_t':
                patchlet_variance = torch.linalg.norm(
                    torch.var(torch.mean(patchlet_points.reshape(b, t, n, self.k, d), -2), 1), dim=-1)
            else:
                raise ValueError("downsample method not supported ")
            _, selected_idxs = torch.topk(patchlet_variance, self.npoints)
            selected_idxs = selected_idxs.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints)
            patchlet_points = utils.index_points(patchlet_points, selected_idxs)
            patchlet_feats = utils.index_points(patchlet_feats, selected_idxs)
            distances = utils.index_points(distances, selected_idxs)
            idxs = utils.index_points(idxs, selected_idxs)
            patchlets = utils.index_points(patchlets, selected_idxs)
            n = self.npoints

        return patchlet_points, patchlet_feats, distances, idxs, patchlets, n, d_feat, fps_idx, out_x

    def forward(self, point_seq, feat_seq=None):
        b, t, n, d = point_seq.shape
        n_original = n
        n_out = n

        # forward patchlets
        x1 = point_seq
        x2 = torch.cat([point_seq[:, [0]], point_seq], dim=1)[:, :-1]
        patchlet_points1, patchlet_feats1, distances1, idxs1, patchlets1, n, d_feat, fps_idx, out_x1 = \
            self.get_tpatches(x1, x2, feat_seq, flip=False)

        #backward patchlets
        x1 = torch.flip(point_seq, [1])
        x2 = torch.cat([x1[:, [0]], x1], dim=1)[:, :-1]
        patchlet_points2, patchlet_feats2, distances2, idxs2, patchlets2, _, _, fps_idx2, out_x2 = \
            self.get_tpatches(x1, x2, feat_seq, flip=True)

        # randomly select a subset
        rand_idxs = torch.randperm(n)[:int(n/2)]
        patchlets = torch.concat([patchlets1[:, rand_idxs, :], patchlets2[:, rand_idxs, :]], 1)
        patchlet_feats = torch.concat([patchlet_feats1[:, rand_idxs, :], patchlet_feats2[:, rand_idxs, :]], 1)
        patchlet_points = torch.concat([patchlet_points1[:, rand_idxs, :], patchlet_points2[:, rand_idxs, :]], 1)
        distances = torch.concat([distances1[:, rand_idxs, :], distances2[:, rand_idxs, :]], 1)
        idxs = torch.concat([idxs1[:, rand_idxs, :], idxs2[:, rand_idxs, :]], 1)
        out_x = out_x1 # remove after debug, out_x is unused

        # reshape all to bxtxnxk
        distances, idxs = distances.reshape(b, t, n, self.k), idxs.reshape(b, t, n, self.k)
        patchlets, patchlet_points = patchlets.reshape(b, t, n, self.k), patchlet_points.reshape(b, t, n, self.k, d)
        patchlet_feats = patchlet_feats.reshape(b, t, n, self.k, d_feat)

        trajectetories = patchlet_points[:, :, :, [0], :]
        normalized_patchlet_points = patchlet_points - trajectetories # normalize the patchlet around the center point of the first frame
        trajectetories = trajectetories - patchlet_points[:, 0, :, [0], :].unsqueeze(1)
        patchlet_feats = torch.cat([patchlet_feats, normalized_patchlet_points], -1)

        return {'idx': idxs, 'distances': distances, 'patchlets': patchlets,
                'patchlet_points': patchlet_points, 'patchlet_feats': patchlet_feats,
                'normalized_patchlet_points': normalized_patchlet_points, 'fps_idx': fps_idx,
                'x_current': out_x.reshape(b, t, n_out, d),
                'trajectories': trajectetories.squeeze(3)}


class PatchletTemporalConv(nn.Module):
    def __init__(self, in_channel, temporal_conv, mlp, use_attn=False, attn_num_heads=4, temporal_stride=1):
        super(PatchletTemporalConv, self).__init__()
        self.use_attn = use_attn
        self.temporal_stride = temporal_stride
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv3d(last_channel, out_channel, 1))
            # self.mlp_convs.append(nn.Conv3d(last_channel, out_channel, [1, temporal_conv, k], 1, padding='same'))
            self.mlp_bns.append(nn.BatchNorm3d(out_channel))
            last_channel = out_channel

        if self.use_attn:
            self.multihead_attn = nn.MultiheadAttention(last_channel, attn_num_heads, batch_first=True)
        else:
            self.temporal_conv = nn.Conv2d(out_channel, out_channel, [1, temporal_conv], [1, temporal_stride])
            self.bnt = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        b, d, n, t, k = x.shape
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x = F.relu(bn(conv(x)))

        x = torch.max(x, -1)[0] # pool neighbors to get patch representation
        # x = torch.mean(x, -1)  # pool neighbors to get patch representation

        if self.use_attn:
            x = x.permute(0, 2, 3, 1)
            b, n, f, c = x.shape
            x = x.reshape(b * n, f, c)
            attn, _ = self.multihead_attn(x, x, x)
            x = attn.view(b, n, f, c).permute(0, 2, 1, 3)
        else:
            x = F.relu(self.bnt(self.temporal_conv(x))) # convolve temporally to improve patch representation
            if not self.temporal_stride == 1:
                x = x.repeat(1, 1, 1, int(t/x.shape[-1]))
            x = x.permute(0, 3, 2, 1)
        return x

class PointMLP(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointMLP, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, x):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x = F.relu(bn(conv(x)))
        return x.permute(0, 2, 3, 1)



class tPatchTraj(nn.Module):
    def __init__(self, model_cfg, num_class, n_frames=32, in_channel=3):
        super(tPatchTraj, self).__init__()
        cfg = model_cfg['PATCHLET']
        self.k_list = cfg.get('k', [16, 16, 16])
        self.sample_mode = cfg['sample_mode']
        self.centroid_jitter = cfg['centroid_jitter']
        self.n_frames = n_frames
        self.downsample_method = cfg['downsample_method']
        self.radius = cfg['radius']
        self.type = cfg.get('type', 'origin')
        attn_num_heads = cfg.get('attn_num_heads')
        self.in_channel = in_channel
        self.extractor_type = cfg.get('extractor_type', 'bidirectional')
        self.temporal_stride = cfg.get('temporal_stride', 1)
        self.local_temp_convs = cfg.get('local_temp_convs', [8, 8, 8])
        npoints = cfg.get('npoints', [512, 128, None])
        temp_conv = cfg.get('temp_conv', n_frames)

        if self.extractor_type == 'bidirectional':
            Extractor = PatchletsExtractorBidirectional
        elif self.extractor_type == 'strided':
            Extractor = PatchletsExtractorStrided
        elif self.extractor_type == 'vanilla':
            Extractor = PatchletsExtractor
        else:
            raise ValueError("Unsupported extractor type")

        use_attn = False
        if self.type == 'attn_all_layers':
            use_attn = True

        # self.point_mlp = PointMLP(in_channel=in_channel, mlp=[64, 64, 128])
        self.patchlet_extractor1 = Extractor(k=self.k_list[0], sample_mode=self.sample_mode, npoints=npoints[0],
                                             add_centroid_jitter=self.centroid_jitter,
                                             downsample_method=self.downsample_method, radius=self.radius[0],
                                             temporal_stride=self.temporal_stride)
        self.patchlet_temporal_conv1 = PatchletTemporalConv(in_channel=in_channel,
                                                            temporal_conv=self.local_temp_convs[0], mlp=[64, 64, 128],
                                                            use_attn=use_attn, attn_num_heads=attn_num_heads,
                                                            temporal_stride=self.temporal_stride)
        self.traj_mlp1 = PointMLP(3, [64, 64, 128])
        self.patchlet_extractor2 = Extractor(k=self.k_list[1], sample_mode=self.sample_mode, npoints=npoints[1],
                                             add_centroid_jitter=self.centroid_jitter,
                                             downsample_method=self.downsample_method, radius=self.radius[1],
                                             temporal_stride=self.temporal_stride)
        self.patchlet_temporal_conv2 = PatchletTemporalConv(in_channel=128+3+128, temporal_conv=self.local_temp_convs[1],
                                                            mlp=[128, 128, 256],
                                                            use_attn=use_attn, attn_num_heads=attn_num_heads,
                                                            temporal_stride=self.temporal_stride)
        self.traj_mlp2 = PointMLP(3, [64, 64, 128])
        self.patchlet_extractor3 = Extractor(k=self.k_list[2], sample_mode=self.sample_mode, npoints=npoints[2],
                                             add_centroid_jitter=self.centroid_jitter,
                                             downsample_method=None, radius=self.radius[2],
                                             temporal_stride=self.temporal_stride)
        self.patchlet_temporal_conv3 = PatchletTemporalConv(in_channel=256+3+128, temporal_conv=self.local_temp_convs[2],
                                                            mlp=[256, 512, 1024],
                                                            use_attn=use_attn, attn_num_heads=attn_num_heads,
                                                            temporal_stride=self.temporal_stride)
        self.traj_mlp3 = PointMLP(3, [64, 64, 128])
        # self.temporal_pool = torch.nn.MaxPool3d([n_frames, 1, 1])
        # self.temporal_pool = torch.nn.AvgPool2d(3, stride=1, padding=1)

        self.fc1 = nn.Linear(1024+128, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

        # self.bnt = nn.BatchNorm1d(1024)
        # self.temporalconv1 = torch.nn.Conv1d(1024, 1024, int(n_frames/4), 1, padding='same')
        self.temporalconv2 = torch.nn.Conv1d(256, 256, temp_conv, 1, padding='same')
        self.bn3 = nn.BatchNorm1d(256)

        self.strided_maxpool = torch.nn.MaxPool2d([1, self.temporal_stride], stride=[1, self.temporal_stride])

        if self.type == 'attn_last_layer' or self.type == 'attn_all_layers':
            embed_dim = 256
            self.multihead_attn = nn.MultiheadAttention(embed_dim, attn_num_heads, batch_first=True)

    def forward(self, xyz):
        b, t, d, n = xyz.shape

        patchlet_dict = self.patchlet_extractor1(xyz.permute(0, 1, 3, 2))
        xyz0 = patchlet_dict['patchlet_points']
        patchlet_feats = patchlet_dict['normalized_patchlet_points'].permute(0, 4, 2, 1, 3)
        patchlet_feats = self.patchlet_temporal_conv1(patchlet_feats)  # [b, d+k, npoint, t, nsample]
        trajectories = patchlet_dict['trajectories']
        traj_features = self.traj_mlp1(trajectories.permute(0, 3, 1, 2))
        # traj_features = torch.max(traj_features, 1)[0].unsqueeze(1).repeat(1, t, 1, 1)
        traj_features = self.strided_maxpool(traj_features.permute(0, 2, 3, 1))
        traj_features = traj_features.repeat_interleave(int(t/traj_features.shape[-1]), dim=-1).permute(0, 3, 1, 2)  # btnc
        patchlet_feats = torch.cat([patchlet_feats, traj_features], -1)

        patchlet_dict = self.patchlet_extractor2(xyz0[:, :, :, 0, :], patchlet_feats)
        xyz = patchlet_dict['patchlet_points']
        patchlet_feats = patchlet_dict['patchlet_feats'].permute(0, 4, 2, 1, 3)
        patchlet_feats = self.patchlet_temporal_conv2(patchlet_feats)  # [b, d+k, npoint, t, nsample]
        trajectories = patchlet_dict['trajectories']
        traj_features = self.traj_mlp2(trajectories.permute(0, 3, 1, 2))
        traj_features = self.strided_maxpool(traj_features.permute(0, 2, 3, 1))
        traj_features = traj_features.repeat_interleave(int(t/traj_features.shape[-1]), dim=-1).permute(0, 3, 1, 2)  # btnc

        # traj_features = torch.max(traj_features, 1)[0].unsqueeze(1).repeat(1, t, 1, 1)
        patchlet_feats = torch.cat([patchlet_feats, traj_features], -1)

        patchlet_dict = self.patchlet_extractor3(xyz[:, :, :, 0, :], patchlet_feats)
        xyz = patchlet_dict['patchlet_points']
        patchlet_feats = patchlet_dict['patchlet_feats'].permute(0, 4, 2, 1, 3)
        patchlet_feats = self.patchlet_temporal_conv3(patchlet_feats)  # [b, d+k, npoint, t, nsample]
        trajectories = patchlet_dict['trajectories']
        traj_features = self.traj_mlp3(trajectories.permute(0, 3, 1, 2))
        traj_features = self.strided_maxpool(traj_features.permute(0, 2, 3, 1))
        traj_features = traj_features.repeat_interleave(int(t/traj_features.shape[-1]), dim=-1).permute(0, 3, 1, 2)  # btnc
        # traj_features = torch.max(traj_features, 1)[0].unsqueeze(1).repeat(1, t, 1, 1)
        patchlet_feats = torch.cat([patchlet_feats, traj_features], -1)


        xyz, patchlet_feats = xyz.squeeze(-1), patchlet_feats.squeeze(-1)
        x = torch.max(patchlet_feats, -2)[0]
        # x = torch.mean(patchlet_feats, -2)
        x = x.reshape(b*t, 1024+128)

        x = self.drop1(F.relu(self.bn1(self.fc1(x).reshape(b, t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512))
        x = self.drop2(F.relu(self.bn2(self.fc2(x).reshape(b, t, 256).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 256))

        # learn a temporal filter on all per-frame global representations
        if self.type == 'origin':
            x = F.relu(
                self.bn3(self.temporalconv2(x.reshape(b, t, 256).permute(0, 2, 1)).permute(0, 2, 1).reshape(-1, 256)))
        elif self.type == 'skip_con':
            x = x + F.relu(
                self.bn3(self.temporalconv2(x.reshape(b, t, 256).permute(0, 2, 1)).permute(0, 2, 1).reshape(-1, 256)))
        elif self.type == 'attn_last_layer' or self.type == 'attn_all_layers':
            x = x.view(b, t, -1)
            attn_output, _ = self.multihead_attn(x, x, x)
            x = attn_output.reshape(b * t, -1)
        else:
            raise NotImplementedError
        x = self.fc3(x)

        # x = self.temporal_pool(x.reshape(b, t, -1))
        x = F.log_softmax(x, -1)

        return {'pred': x.reshape(b, t, -1).permute([0, 2, 1]), 'features': patchlet_feats,
                'patchlet_points': xyz0}
