# Sinkhorn implementation by Francis Williams: https://github.com/fwilliams/scalable-pytorch-sinkhorn
# adapted to this project's requirements
from typing import Union
from scipy.spatial import KDTree
import pykeops.torch as keops
import torch
import tqdm
import numpy as np

class SinkhornCorr(torch.nn.Module):
    def __init__(self, p_norm=2, max_iters=100, eps=1e-3, stop_thresh=1e-5):
        super(SinkhornCorr, self).__init__()
        self.max_iters = max_iters
        self.eps = eps
        self.stop_thresh = stop_thresh
        self.p_norm = p_norm
    def forward(self, x1, x2):
        distance, corr_mat, max_ind12, max_ind21 = sinkhorn(x1, x2, p_norm=self.p_norm, max_iters=self.max_iters,
                                                            eps=self.eps, stop_thresh=self.stop_thresh)
        return {'out1': x1, 'out2': x2, 'corr_mat': corr_mat, 'corr_idx12': max_ind12, 'corr_idx21': max_ind21}



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
    if squeezed:
        ret = torch.squeeze(ret)

    return ret


def sinkhorn(p, q, p_norm=2, eps=1e-4, max_iters=100, stop_thresh=1e-3):
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
    M = pairwise_distances(p, q, p_norm)#.squeeze()
    a = (torch.ones(p.shape[0:-1], dtype=torch.float32, device=p.device) / p.shape[0])#.squeeze()
    b = (torch.ones(q.shape[0:-1], dtype=torch.float32, device=q.device) / q.shape[0])#.squeeze()

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

    return (P * M).sum(), P, approx_corr_1, approx_corr_2

