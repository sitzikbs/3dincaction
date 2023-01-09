# a pytorch adaptation of https://github.com/fwilliams/point-cloud-utils/blob/8aef71387d56a56398e1b0482e268f972d306d34/point_cloud_utils/_sinkhorn.py

from scipy.spatial import KDTree
import torch
import tqdm
import numpy as np
import torch.nn as nn

import numpy as np


def pairwise_distances(a, b, p=None):
    """
    Compute the (batched) pairwise distance matrix between a and b which both have size [m, n, d] or [n, d]. The result is a tensor of size [m, n, n] (or [n, n]) whose entry [m, i, j] contains the distance_tensor between a[m, i, :] and b[m, j, :].
    Args:
      a : A tensor containing m batches of n points of dimension d. i.e. of size (m, n, d)
      b : A tensor containing m batches of n points of dimension d. i.e. of size (m, n, d)
      p : Norm to use for the distance_tensor
    Returns:
      M : A (m, n, n)-shaped array containing the pairwise distance_tensor between each pair of inputs in a batch.
    """

    squeezed = False
    if len(a.shape) == 2 and len(b.shape) == 2:
        a = a[np.newaxis, :, :]
        b = b[np.newaxis, :, :]
        squeezed = True

    if len(a.shape) != 3:
        raise ValueError("Invalid shape for a. Must be [m, n, d] or [n, d] but got", a.shape)
    if len(b.shape) != 3:
        raise ValueError("Invalid shape for a. Must be [m, n, d] or [n, d] but got", b.shape)

    ret = torch.linalg.norm(a[:, :, np.newaxis, :] - b[:, np.newaxis, :, :], axis=-1, ord=p)
    # ret = np.power(np.abs(a[:, :, np.newaxis, :] - b[:, np.newaxis, :, :]), p).sum(3)
    if squeezed:
        ret = torch.squeeze(ret)

    return ret


def sinkhorn(a, b, M, eps, max_iters=100, stop_thresh=1e-3):
    """
    Compute the (batched) Sinkhorn correspondences between two dirac delta distributions, U, and V.
    This implementation is numerically stable with float32.
    Args:
      a : A m-sized minibatch of weights for each dirac in the first distribution, U. i.e. shape = (m, n)
      b : A m-sized minibatch of weights for each dirac in the second distribution, V. i.e. shape = (m, n)
      M : A minibatch of n-by-n tensors storing the distance between each pair of diracs in U and V. i.e. shape = (m, n, n) and each i.e. M[k, i, j] = ||u[k,_i] - v[k, j]||
      eps : The reciprocal of the sinkhorn regularization parameter
      max_iters : The maximum number of Sinkhorn iterations
      stop_thresh : Stop if the change in iterates is below this value
    Returns:
      P : An (m, n, n)-shaped array of correspondences between distributions U and V
    """
    # a and b are tensors of size [nb, m] and [nb, n]
    # M is a tensor of size [nb, m, n]

    M = torch.squeeze(M)
    a = torch.squeeze(a)
    b = torch.squeeze(b)
    squeezed = False

    if len(M.shape) == 2 and len(a.shape) == 1 and len(b.shape) == 1:
        M = M[np.newaxis, :, :]
        a = a[np.newaxis, :]
        b = b[np.newaxis, :]
        squeezed = True
    elif len(M.shape) == 2 and len(a.shape) != 1:
        raise ValueError("Invalid shape for a %s, expected [m,] where m is the number of samples in a and "
                         "M has shape [m, n]" % str(a.shape))
    elif len(M.shape) == 2 and len(b.shape) != 1:
        raise ValueError("Invalid shape for a %s, expected [m,] where n is the number of samples in a and "
                         "M has shape [m, n]" % str(b.shape))

    if len(M.shape) != 3:
        raise ValueError("Got unexpected shape for M %s, should be [nb, m, n] where nb is batch size, and "
                         "m and n are the number of samples in the two input measures." % str(M.shape))
    elif len(M.shape) == 3 and len(a.shape) != 2:
        raise ValueError("Invalid shape for a %s, expected [nb, m]  where nb is batch size, m is the number of samples "
                         "in a and M has shape [nb, m, n]" % str(a.shape))
    elif len(M.shape) == 3 and len(b.shape) != 2:
        raise ValueError("Invalid shape for a %s, expected [nb, m]  where nb is batch size, m is the number of samples "
                         "in a and M has shape [nb, m, n]" % str(b.shape))

    nb = M.shape[0]
    m = M.shape[1]
    n = M.shape[2]

    if a.dtype != b.dtype or a.dtype != M.dtype:
        raise ValueError("Tensors a, b, and M must have the same dtype got: dtype(a) = %s, dtype(b) = %s, dtype(M) = %s"
                         % (str(a.dtype), str(b.dtype), str(M.dtype)))
    if a.shape != (nb, m):
        raise ValueError("Got unexpected shape for tensor a (%s). Expected [nb, m] where M has shape [nb, m, n]." %
                         str(a.shape))
    if b.shape != (nb, n):
        raise ValueError("Got unexpected shape for tensor b (%s). Expected [nb, n] where M has shape [nb, m, n]." %
                         str(b.shape))

    u = torch.zeros_like(a)
    v = torch.zeros_like(b)

    M_t = torch.permute(M, (0, 2, 1))

    # def stabilized_log_sum_exp(x):
    #     max_x = x.max(2)
    #     x = x - max_x[:, :, np.newaxis]
    #     ret = torch.log(torch.sum(torch.exp(x), axis=2)) + max_x
    #     ret = torch.logsumexp(x, axis=2) + max_x
    #     return ret

    for current_iter in range(max_iters):
        u_prev = u
        v_prev = v

        summand_u = (-M + v.unsqueeze(1)) / eps
        u = eps * (torch.log(a) - torch.logsumexp(summand_u, 2))

        summand_v = (-M_t + u.unsqueeze(1)) / eps
        v = eps * (torch.log(b) - torch.logsumexp(summand_v, 2))

        err_u = torch.sum(torch.abs(u_prev-u), axis=1).max()
        err_v = torch.sum(torch.abs(v_prev-v), axis=1).max()

        if err_u < stop_thresh and err_v < stop_thresh:
            break

    log_P = (-M + u.unsqueeze(2) + v.unsqueeze(1)) / eps

    P = torch.exp(log_P)

    if squeezed:
        P = torch.squeeze(P)

    approx_corr_1 = P.argmax(dim=-1).squeeze(-1)
    approx_corr_2 = P.argmax(dim=-2).squeeze(-1)

    return P, approx_corr_1, approx_corr_2


def earth_movers_distance(p, q, p_norm=2, eps=1e-4, max_iters=100, stop_thresh=1e-3):
    """
    Compute the (batched) Sinkhorn correspondences between two dirac delta distributions, U, and V.
    This implementation is numerically stable with float32.
    Args:
      p : An (n, d)-shaped array of d-dimensional points
      b : An (m, d)-shaped array of d-dimensional points
      p_norm : Which norm to use. Must be one of {non-zero int, inf, -inf, ‘fro’, ‘nuc’} (default is 2),
      eps : The reciprocal of the sinkhorn regularization parameter (default 1e-4)
      max_iters : The maximum number of Sinkhorn iterations
      stop_thresh : Stop if the change in iterates is below this value
    Returns:
      emd : The earth mover's distance between point clouds p and q
      P : An (n, m)-shaped array of correspondences between point clouds p and q
    """

    M = pairwise_distances(p, q, p_norm)
    a = torch.ones(p.shape[0:-1], dtype=torch.float32, device=p.device) / p.shape[0]
    b = torch.ones(q.shape[0:-1], dtype=torch.float32, device=q.device) / q.shape[0]
    P, approx_corr_1, approx_corr_2 = sinkhorn(a, b, M, eps, max_iters, stop_thresh)

    return (P * M).sum(), P, approx_corr_1, approx_corr_2

class NoiseGenerator(torch.utils.data.Dataset):
    def __init__(self, n_points, radius=0.5, n_samples=1, sigma=0.3):
        # # Generate random spherical coordinates

        self.radius = radius
        self.n_points = n_points
        self.points = []
        for i in range(n_samples):
            self.points.append(self.get_noisy_points(sigma))
    def get_noisy_points(self, sigma):
        #TODO add local distortion (use kdtree and fixed motion vec)
        return np.clip(np.random.randn(self.n_points, 3)*sigma, -1, 1).astype(np.float32)

    def __len__(self):
        return len(self.points)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return {'points': self.points[idx]}

def local_distort(points, r=0.1, ratio=0.15, sigma=0.05):
    b, n, _ = points.size()
    n_ratio = int(ratio*n)

    points = points.cpu().numpy()
    subset = torch.randperm(n)[:b]
    translation_vec = np.random.rand(b, 3) * sigma

    for i, pts in enumerate(points):
        tree = KDTree(pts)
        _, nn_idx = tree.query(points[i, subset[i], :], k=n_ratio) #distort knn
        points[i, nn_idx, :] += translation_vec[i]

    return torch.tensor(points)


if __name__ == "__main__":
    batch_size = 8
    n_points = 1024
    n_epochs = 10000

    dataset = NoiseGenerator(n_points, n_samples=10000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                  pin_memory=True)
    for epoch in range(n_epochs):
        for batch_ind, data in enumerate(dataloader):
            print("Epoch {}, Batch {}".format(epoch, batch_ind))
            points1 = data['points'].cuda()
            # points2 = local_distort(data['points']).cuda()
            points2 = points1
            with torch.no_grad():
                output = earth_movers_distance(points1.squeeze(), points2.squeeze())