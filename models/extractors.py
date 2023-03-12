import faiss
from faiss.contrib.torch_utils import torch_replacement_knn_gpu as faiss_torch_knn_gpu
from pykeops.torch import Vi, Vj
from scipy.spatial import cKDTree
import numpy as np
import torch
import torch.nn as nn
import models.pointnet2_utils as utils


def get_tpatches(x1, x2, feat_seq, flip=False, k=16, radius=0.2, res=faiss.StandardGpuResources(),
                 sample_mode='nn', add_centroid_jitter=None, downsample_method='fps', npoints=None):
    b, t, n, d = x1.shape
    n_out = n

    if feat_seq is None:
        feat_seq = x1
        d_feat = d
    else:
        d_feat = feat_seq.shape[-1]

    out_x = torch.empty(b, t, n_out, d)
    patchlets = torch.empty(b, t, n, k, device=x1.device, dtype=torch.long)
    distances_i = torch.empty(b, t, n, k, device=x1.device)
    idxs_i = torch.empty(b, t, n, k, device=x1.device, dtype=torch.long)
    patchlet_points = torch.empty(b, t, n, k, d, device=x1.device)
    patchlet_feats = torch.empty(b, t, n, k, d_feat, device=x1.device)

    # loop over the data to reorder the indices to form the patchlets
    x_current = x2[:, 0]
    feat_seq_2 = torch.cat([feat_seq[:, [0]], feat_seq], dim=1)[:, :-1]
    for i in range(0, t):
        x_next = x1[:, i]
        distances, idxs = get_knn(x_current[..., 0:3], x_next[..., 0:3], k=k, res=res, method='keops',
                                  radius=radius)
        if sample_mode == 'nn':
            x_current = utils.index_points(x_next, idxs)[:, :, 0, :]
        elif sample_mode == 'randn':
            rand_idx = torch.randint(k, (b, n, 1), device=x_next.device, dtype=torch.int64).repeat(
                [1, 1, 3]).unsqueeze(2)
            x_current = torch.gather(utils.index_points(x_next, idxs).squeeze(), dim=2, index=rand_idx).squeeze()
        elif sample_mode == 'gt':
            # only works when point correspondence is known and points are already aligned
            x_current = x_next
        else:
            raise ValueError("sample mode not supported")

        out_x[:, i] = x_current
        if add_centroid_jitter is not None and not sample_mode == 'gt':
            x_current = x_current + add_centroid_jitter * torch.randn_like(x_current)

        distances_i[:, i], idxs_i[:, i] = distances, idxs
        patchlets[:, i] = idxs_i[:, i]
        patchlet_points[:, i] = utils.index_points(x_next, idxs).squeeze()
        patchlet_feats[:, i] = utils.index_points(feat_seq_2[:, i], idxs).squeeze()

    distances = distances_i
    idxs = idxs_i
    if flip:  # reverse temporal order
        patchlet_feats = torch.flip(patchlet_feats, [1])
        patchlet_points = torch.flip(patchlet_points, [1])
        idxs = torch.flip(idxs, [1])
        distances = torch.flip(distances, [1])
        patchlets = torch.flip(patchlets, [1])

    patchlet_feats = patchlet_feats.reshape(b * t, n, k, d_feat)
    patchlet_points = patchlet_points.reshape(b * t, n, k, d)
    idxs = idxs.reshape(b * t, n, k)
    distances = distances.reshape(b * t, n, k)
    patchlets = patchlets.reshape(b * t, n, k)

    fps_idx = None
    if downsample_method == 'fps':
        # select a subset of the points using fps for maximum coverage
        selected_idxs = utils.farthest_point_sample(x1[:, 0].contiguous(), npoints).to(torch.int64)
        selected_idxs = selected_idxs.unsqueeze(1).repeat([1, t, 1]).reshape(-1, npoints)
        patchlet_points = utils.index_points(patchlet_points, selected_idxs)
        patchlet_feats = utils.index_points(patchlet_feats, selected_idxs)
        distances = utils.index_points(distances, selected_idxs)
        idxs = utils.index_points(idxs, selected_idxs)
        patchlets = utils.index_points(patchlets, selected_idxs)
        n = npoints
    elif downsample_method == 'var' or downsample_method == 'mean_var_t':
        # select a subset of the points with the largest point variance for maximum temporal movement
        if downsample_method == 'var':
            temporal_patchlet_points = patchlet_points.reshape(b, t, n, k, d). \
                permute(0, 2, 1, 3, 4).reshape(b, n, -1, d)
            patchlet_variance = torch.linalg.norm(torch.var(temporal_patchlet_points, -2), dim=-1)
        elif downsample_method == 'mean_var_t':
            patchlet_variance = torch.linalg.norm(
                torch.var(torch.mean(patchlet_points.reshape(b, t, n, k, d), -2), 1), dim=-1)
        else:
            raise ValueError("downsample method not supported ")
        _, selected_idxs = torch.topk(patchlet_variance, npoints)
        selected_idxs = selected_idxs.unsqueeze(1).repeat([1, t, 1]).reshape(-1, npoints)
        patchlet_points = utils.index_points(patchlet_points, selected_idxs)
        patchlet_feats = utils.index_points(patchlet_feats, selected_idxs)
        distances = utils.index_points(distances, selected_idxs)
        idxs = utils.index_points(idxs, selected_idxs)
        patchlets = utils.index_points(patchlets, selected_idxs)
        n = npoints

    return patchlet_points, patchlet_feats, distances, idxs, patchlets, n, d_feat, fps_idx, out_x


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
                 radius=None, temporal_stride=8):
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
                x_current = x_next
            else:
                raise ValueError("sample mode not supported")

            out_x[:, i] = x_current
            if self.add_centroid_jitter is not None and not self.sample_mode == 'gt':
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



class PatchletsExtractorBidirectional(nn.Module):
    def __init__(self, k=16, sample_mode='nn', npoints=None, add_centroid_jitter=None, downsample_method=None,
                 radius=None, temporal_stride=8):
        super(PatchletsExtractorBidirectional, self).__init__()
        self.k = k
        self.radius = radius
        self.sample_mode = sample_mode
        self.downsample_method = downsample_method
        self.npoints = npoints
        self.add_centroid_jitter = add_centroid_jitter
        self.res = faiss.StandardGpuResources()
        self.res.setDefaultNullStreamAllDevices()

    # def get_tpatches(self, x1, x2, feat_seq, flip=False):
    #
    #     b, t, n, d = x1.shape
    #     n_out = n
    #
    #     if feat_seq is None:
    #         feat_seq = x1
    #         d_feat = d
    #     else:
    #         d_feat = feat_seq.shape[-1]
    #
    #     out_x = torch.empty(b, t, n_out, d)
    #     patchlets = torch.empty(b, t, n, self.k, device=x1.device, dtype=torch.long)
    #     distances_i = torch.empty(b, t, n, self.k, device=x1.device)
    #     idxs_i = torch.empty(b, t, n, self.k, device=x1.device, dtype=torch.long)
    #     patchlet_points = torch.empty(b, t, n, self.k, d, device=x1.device)
    #     patchlet_feats = torch.empty(b, t, n, self.k, d_feat, device=x1.device)
    #
    #     # loop over the data to reorder the indices to form the patchlets
    #     x_current = x2[:, 0]
    #     feat_seq_2 = torch.cat([feat_seq[:, [0]], feat_seq], dim=1)[:, :-1]
    #     for i in range(0, t):
    #         x_next = x1[:, i]
    #         distances, idxs = get_knn(x_current[..., 0:3], x_next[..., 0:3], k=self.k, res=self.res, method='keops',
    #                                   radius=self.radius)
    #         if self.sample_mode == 'nn':
    #             x_current = utils.index_points(x_next, idxs)[:, :, 0, :]
    #         elif self.sample_mode == 'randn':
    #             rand_idx = torch.randint(self.k, (b, n, 1), device=x_next.device, dtype=torch.int64).repeat(
    #                 [1, 1, 3]).unsqueeze(2)
    #             x_current = torch.gather(utils.index_points(x_next, idxs).squeeze(), dim=2, index=rand_idx).squeeze()
    #         elif self.sample_mode == 'gt':
    #             # only works when point correspondence is known and points are already aligned
    #             # if self.downsample_method == 'fps':
    #             #     x_current = utils.index_points(x_next, fps_idx).contiguous()
    #             # else:
    #             x_current = x_next
    #         else:
    #             raise ValueError("sample mode not supported")
    #
    #         out_x[:, i] = x_current
    #         if self.add_centroid_jitter is not None  and not self.sample_mode == 'gt':
    #             x_current = x_current + self.add_centroid_jitter * torch.randn_like(x_current)
    #
    #         distances_i[:, i], idxs_i[:, i] = distances, idxs
    #         patchlets[:, i] = idxs_i[:, i]
    #         patchlet_points[:, i] = utils.index_points(x_next, idxs).squeeze()
    #         patchlet_feats[:, i] = utils.index_points(feat_seq_2[:, i], idxs).squeeze()
    #
    #     distances = distances_i
    #     idxs = idxs_i
    #     if flip: # reverse temporal order
    #         patchlet_feats = torch.flip(patchlet_feats, [1])
    #         patchlet_points = torch.flip(patchlet_points, [1])
    #         idxs = torch.flip(idxs, [1])
    #         distances = torch.flip(distances, [1])
    #         patchlets = torch.flip(patchlets, [1])
    #
    #     patchlet_feats = patchlet_feats.reshape(b * t, n, self.k, d_feat)
    #     patchlet_points = patchlet_points.reshape(b * t, n, self.k, d)
    #     idxs = idxs.reshape(b * t, n, self.k)
    #     distances = distances.reshape(b * t, n, self.k)
    #     patchlets = patchlets.reshape(b * t, n, self.k)
    #
    #     fps_idx = None
    #     if self.downsample_method == 'fps':
    #         # select a subset of the points using fps for maximum coverage
    #         selected_idxs = utils.farthest_point_sample(x1[:, 0].contiguous(), self.npoints).to(torch.int64)
    #         selected_idxs = selected_idxs.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints)
    #         patchlet_points = utils.index_points(patchlet_points, selected_idxs)
    #         patchlet_feats = utils.index_points(patchlet_feats, selected_idxs)
    #         distances = utils.index_points(distances, selected_idxs)
    #         idxs = utils.index_points(idxs, selected_idxs)
    #         patchlets = utils.index_points(patchlets, selected_idxs)
    #         n = self.npoints
    #     elif self.downsample_method == 'var' or self.downsample_method == 'mean_var_t':
    #         # select a subset of the points with the largest point variance for maximum temporal movement
    #         if self.downsample_method == 'var':
    #             temporal_patchlet_points = patchlet_points.reshape(b, t, n, self.k, d).\
    #                 permute(0, 2, 1, 3, 4).reshape(b, n,-1, d)
    #             patchlet_variance = torch.linalg.norm(torch.var(temporal_patchlet_points, -2), dim=-1)
    #         elif self.downsample_method == 'mean_var_t':
    #             patchlet_variance = torch.linalg.norm(
    #                 torch.var(torch.mean(patchlet_points.reshape(b, t, n, self.k, d), -2), 1), dim=-1)
    #         else:
    #             raise ValueError("downsample method not supported ")
    #         _, selected_idxs = torch.topk(patchlet_variance, self.npoints)
    #         selected_idxs = selected_idxs.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints)
    #         patchlet_points = utils.index_points(patchlet_points, selected_idxs)
    #         patchlet_feats = utils.index_points(patchlet_feats, selected_idxs)
    #         distances = utils.index_points(distances, selected_idxs)
    #         idxs = utils.index_points(idxs, selected_idxs)
    #         patchlets = utils.index_points(patchlets, selected_idxs)
    #         n = self.npoints
    #
    #     return patchlet_points, patchlet_feats, distances, idxs, patchlets, n, d_feat, fps_idx, out_x

    def forward(self, point_seq, feat_seq=None):
        b, t, n, d = point_seq.shape
        n_original = n
        n_out = n

        # forward patchlets
        x1 = point_seq
        x2 = torch.cat([point_seq[:, [0]], point_seq], dim=1)[:, :-1]
        patchlet_points1, patchlet_feats1, distances1, idxs1, patchlets1, n, d_feat, fps_idx, out_x1 = \
            get_tpatches(x1, x2, feat_seq, flip=False,
                              k=self.k, radius=self.radius, res=self.res, sample_mode=self.sample_mode,
                              add_centroid_jitter=self.add_centroid_jitter,
                              downsample_method=self.downsample_method, npoints=self.npoints )

        #backward patchlets
        x1 = torch.flip(point_seq, [1])
        x2 = torch.cat([x1[:, [0]], x1], dim=1)[:, :-1]
        patchlet_points2, patchlet_feats2, distances2, idxs2, patchlets2, _, _, fps_idx2, out_x2 = \
            get_tpatches(x1, x2, feat_seq, flip=True,
                              k=self.k, radius=self.radius, res=self.res, sample_mode=self.sample_mode,
                              add_centroid_jitter=self.add_centroid_jitter,
                              downsample_method=self.downsample_method, npoints=self.npoints )

        # randomly select a subset
        rand_idxs = torch.randperm(n)[:int(n/2)]
        patchlets = torch.cat([patchlets1[:, rand_idxs, :], patchlets2[:, rand_idxs, :]], 1)
        patchlet_feats = torch.cat([patchlet_feats1[:, rand_idxs, :], patchlet_feats2[:, rand_idxs, :]], 1)
        patchlet_points = torch.cat([patchlet_points1[:, rand_idxs, :], patchlet_points2[:, rand_idxs, :]], 1)
        distances = torch.cat([distances1[:, rand_idxs, :], distances2[:, rand_idxs, :]], 1)
        idxs = torch.cat([idxs1[:, rand_idxs, :], idxs2[:, rand_idxs, :]], 1)
        out_x = out_x1 # remove after debug, out_x is unused

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
        self.k = k
        self.radius = radius
        self.sample_mode = sample_mode
        self.downsample_method = downsample_method
        self.npoints = npoints
        self.add_centroid_jitter = add_centroid_jitter
        self.res = faiss.StandardGpuResources()
        self.res.setDefaultNullStreamAllDevices()
        self.temporal_stride = temporal_stride

    # def get_tpatches(self, x1, x2, feat_seq, flip=False):
    #
    #     b, t, n, d = x1.shape
    #     n_out = n
    #
    #     if feat_seq is None:
    #         feat_seq = x1
    #         d_feat = d
    #     else:
    #         d_feat = feat_seq.shape[-1]
    #
    #     out_x = torch.empty(b, t, n_out, d)
    #     patchlets = torch.empty(b, t, n, self.k, device=x1.device, dtype=torch.long)
    #     distances_i = torch.empty(b, t, n, self.k, device=x1.device)
    #     idxs_i = torch.empty(b, t, n, self.k, device=x1.device, dtype=torch.long)
    #     patchlet_points = torch.empty(b, t, n, self.k, d, device=x1.device)
    #     patchlet_feats = torch.empty(b, t, n, self.k, d_feat, device=x1.device)
    #
    #     # loop over the data to reorder the indices to form the patchlets
    #     x_current = x2[:, 0]
    #     feat_seq_2 = torch.cat([feat_seq[:, [0]], feat_seq], dim=1)[:, :-1]
    #     for i in range(0, t):
    #         x_next = x1[:, i]
    #         distances, idxs = get_knn(x_current[..., 0:3], x_next[..., 0:3], k=self.k, res=self.res, method='keops',
    #                                   radius=self.radius)
    #         if self.sample_mode == 'nn':
    #             x_current = utils.index_points(x_next, idxs)[:, :, 0, :]
    #         elif self.sample_mode == 'randn':
    #             rand_idx = torch.randint(self.k, (b, n, 1), device=x_next.device, dtype=torch.int64).repeat(
    #                 [1, 1, 3]).unsqueeze(2)
    #             x_current = torch.gather(utils.index_points(x_next, idxs).squeeze(), dim=2, index=rand_idx).squeeze()
    #         elif self.sample_mode == 'gt':
    #             # only works when point correspondence is known and points are already aligned
    #             # if self.downsample_method == 'fps':
    #             #     x_current = utils.index_points(x_next, fps_idx).contiguous()
    #             # else:
    #             x_current = x_next
    #         else:
    #             raise ValueError("sample mode not supported")
    #
    #         out_x[:, i] = x_current
    #         if self.add_centroid_jitter is not None  and not self.sample_mode == 'gt':
    #             x_current = x_current + self.add_centroid_jitter * torch.randn_like(x_current)
    #
    #         distances_i[:, i], idxs_i[:, i] = distances, idxs
    #         patchlets[:, i] = idxs_i[:, i]
    #         patchlet_points[:, i] = utils.index_points(x_next, idxs).squeeze()
    #         patchlet_feats[:, i] = utils.index_points(feat_seq_2[:, i], idxs).squeeze()
    #
    #     distances = distances_i
    #     idxs = idxs_i
    #     if flip: # reverse temporal order
    #         patchlet_feats = torch.flip(patchlet_feats, [1])
    #         patchlet_points = torch.flip(patchlet_points, [1])
    #         idxs = torch.flip(idxs, [1])
    #         distances = torch.flip(distances, [1])
    #         patchlets = torch.flip(patchlets, [1])
    #
    #     patchlet_feats = patchlet_feats.reshape(b * t, n, self.k, d_feat)
    #     patchlet_points = patchlet_points.reshape(b * t, n, self.k, d)
    #     idxs = idxs.reshape(b * t, n, self.k)
    #     distances = distances.reshape(b * t, n, self.k)
    #     patchlets = patchlets.reshape(b * t, n, self.k)
    #
    #     fps_idx = None
    #     if self.downsample_method == 'fps':
    #         # select a subset of the points using fps for maximum coverage
    #         selected_idxs = utils.farthest_point_sample(x1[:, 0].contiguous(), self.npoints).to(torch.int64)
    #         selected_idxs = selected_idxs.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints)
    #         patchlet_points = utils.index_points(patchlet_points, selected_idxs)
    #         patchlet_feats = utils.index_points(patchlet_feats, selected_idxs)
    #         distances = utils.index_points(distances, selected_idxs)
    #         idxs = utils.index_points(idxs, selected_idxs)
    #         patchlets = utils.index_points(patchlets, selected_idxs)
    #         n = self.npoints
    #     elif self.downsample_method == 'var' or self.downsample_method == 'mean_var_t':
    #         # select a subset of the points with the largest point variance for maximum temporal movement
    #         if self.downsample_method == 'var':
    #             temporal_patchlet_points = patchlet_points.reshape(b, t, n, self.k, d).\
    #                 permute(0, 2, 1, 3, 4).reshape(b, n,-1, d)
    #             patchlet_variance = torch.linalg.norm(torch.var(temporal_patchlet_points, -2), dim=-1)
    #         elif self.downsample_method == 'mean_var_t':
    #             patchlet_variance = torch.linalg.norm(
    #                 torch.var(torch.mean(patchlet_points.reshape(b, t, n, self.k, d), -2), 1), dim=-1)
    #         else:
    #             raise ValueError("downsample method not supported ")
    #         _, selected_idxs = torch.topk(patchlet_variance, self.npoints)
    #         selected_idxs = selected_idxs.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints)
    #         patchlet_points = utils.index_points(patchlet_points, selected_idxs)
    #         patchlet_feats = utils.index_points(patchlet_feats, selected_idxs)
    #         distances = utils.index_points(distances, selected_idxs)
    #         idxs = utils.index_points(idxs, selected_idxs)
    #         patchlets = utils.index_points(patchlets, selected_idxs)
    #         n = self.npoints
    #
    #     return patchlet_points, patchlet_feats, distances, idxs, patchlets, n, d_feat, fps_idx, out_x

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
                get_tpatches(x1, x2, feats, flip=False,
                              k=self.k, radius=self.radius, res=self.res, sample_mode=self.sample_mode,
                              add_centroid_jitter=self.add_centroid_jitter,
                              downsample_method=self.downsample_method, npoints=self.npoints )

            #bidirectional
            x1 = torch.flip(x1, [1])
            x2 = torch.cat([x1[:, [0]], x1], dim=1)[:, :-1]
            patchlet_points2, patchlet_feats2, distances2, idxs2, patchlets2, _, _, fps_idx2, out_x2 = \
                get_tpatches(x1, x2, feat_seq, flip=True,
                              k=self.k, radius=self.radius, res=self.res, sample_mode=self.sample_mode,
                              add_centroid_jitter=self.add_centroid_jitter,
                              downsample_method=self.downsample_method, npoints=self.npoints )

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

        normalized_patchlet_points = patchlet_points.detach().clone()
        for i in range(n_temporal_segments):
            normalized_patchlet_points[:, i*self.temporal_stride:(i+1)*self.temporal_stride] -= \
                normalized_patchlet_points[:, i * self.temporal_stride, :, [0], :].unsqueeze(1).detach()
        patchlet_feats = torch.cat([patchlet_feats, normalized_patchlet_points], -1)

        return {'idx': idxs, 'distances': distances, 'patchlets': patchlets,
                'patchlet_points': patchlet_points, 'patchlet_feats': patchlet_feats,
                'normalized_patchlet_points': normalized_patchlet_points, 'fps_idx': fps_idx,
                'x_current': out_x.reshape(b, t, n_out, d)}
