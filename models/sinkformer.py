# Adapted from https://github.com/michaelsdr/sinkformers
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
eps = 1


use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
# and from https://github.com/dfdazac/wassdistance/blob/master/layers.py

def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(-2)
    y_lin = y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    return C


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none', cost=cost_matrix):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.cost = cost

    def forward(self, x, y, **kwargs):
        # The Sinkhorn algorithm takes as input three variables :
        C = self.cost(x, y, **kwargs)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False, device=C.device).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False, device=C.device).fill_(1.0 / y_points).squeeze()

        if mu.dim() < 2:
            mu = mu.view(-1, 1)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-2

        # Sinkhorn iterations
        for i in range(self.max_iter):
            if i % 2 == 0:
                u1 = u  # useful to check the update
                u = self.eps * (torch.log(mu) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

                err = (u - u1).abs().sum(-1).mean()
            else:
                v = self.eps * (torch.log(nu) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        return pi, C, U, V

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps



    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

def distmat2(X, Y, div=1):
  X_sq = (X ** 2).sum(axis=-1)
  Y_sq = (Y ** 2).sum(axis=-1)
  cross_term = X.matmul(Y.transpose(1, 2))
  return (X_sq[:, :, None] + Y_sq[:, None, :] - 2 * cross_term) / (div ** 2)

def dotmat(X, Y, div=1):
  return  - X.bmm(Y.transpose(1, 2)) / div


sinkhornkeops = SinkhornDistance(eps=eps, max_iter=1, cost=dotmat)

class MABSINK(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True, sinkhorn=sinkhornkeops):
        super(MABSINK, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.sinkhorn = sinkhorn

    def forward(self, Q, K):
        dim_split = self.dim_V // self.num_heads
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
        sqrtV = math.sqrt(math.sqrt(self.dim_V))

        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        pi, C, U, V = self.sinkhorn(Q_ / sqrtV, K_ / sqrtV)
        A = pi
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class MABSINK_Simplified(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True, sinkhorn=sinkhornkeops):
        super(MABSINK_Simplified, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.sinkhorn = sinkhorn

    def forward(self, Q, K):
        dim_split = self.dim_V // self.num_heads
        Q = self.fc_q(Q)
        sqrtV = math.sqrt(math.sqrt(self.dim_V))
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = Q_
        V_ = Q_
        pi, C, U, V = self.sinkhorn(Q_ / sqrtV, K_ / sqrtV)
        n = Q_.shape[1]
        p = K_.shape[1]
        A = pi * n
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O, A

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        O, A = self.mab(X, X)
        return O, A

class SABSINK(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, sinkhorn=sinkhornkeops):
        super(SABSINK, self).__init__()
        self.mab = MABSINK_Simplified(dim_in, dim_in, dim_out, num_heads, ln=ln, sinkhorn=sinkhorn)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False,
                 save_attn_0=None, save_attn_1=None):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)


    def forward(self, X):
        Y = self.I.repeat(X.size(0), 1, 1)
        H_0, A_0 = self.mab0(Y, X)
        H_1, A_1 = self.mab1(X, H_0)
        return H_1, A_0, A_1

class ISABSINK(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, sinkhorn=sinkhornkeops):
        super(ISABSINK, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MABSINK(dim_out, dim_in, dim_out, num_heads, ln=ln, sinkhorn=sinkhorn)
        self.mab1 = MABSINK(dim_in, dim_out, dim_out, num_heads, ln=ln, sinkhorn=sinkhorn)

    def forward(self, X):

        Y = self.I.repeat(X.size(0), 1, 1)
        H = self.mab0(Y, X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        Z = self.S.repeat(X.size(0), 1, 1)
        pma, _ = self.mab(Z, X)
        return pma

class PMASINK(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, sinkhorn=sinkhornkeops):
        super(PMASINK, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MABSINK(dim, dim, dim, num_heads, ln=ln, sinkhorn=sinkhorn)

    def forward(self, X):
        Z = self.S.repeat(X.size(0), 1, 1)
        return self.mab(Z, X)
class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X


class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads,  ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))

class SetTransformerLegacy(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformerLegacy, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


class ModelNet(nn.Module):
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_output=40,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
        save_attn_0 = 'attn_0.npy',
        save_attn_1 = 'attn_1.npy',
    ):
        super(ModelNet, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln, save_attn_0=None, save_attn_1=None),
            ISAB(dim_hidden, dim_hidden, num_heads,num_inds, ln=ln, save_attn_0=save_attn_0, save_attn_1=save_attn_1),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        Y = self.enc(X)
        return self.dec(Y).squeeze()

class ModelNetSink(nn.Module):
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_output=40,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=True,
        n_it=1
    ):
        super(ModelNetSink, self).__init__()
        sinkhornkeops = SinkhornDistance(eps=eps, max_iter=n_it, cost=dotmat)
        self.enc = nn.Sequential(
            ISABSINK(dim_input, dim_hidden, num_heads, num_inds, ln=ln, sinkhorn=sinkhornkeops),
            ISABSINK(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, sinkhorn=sinkhornkeops),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMASINK(dim_hidden, num_heads, num_outputs, ln=ln, sinkhorn=sinkhornkeops),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        Y = self.enc(X)
        return self.dec(Y).squeeze()

class ModelNetSabSink(nn.Module):
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_output=40,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
        n_it=1,
    ):
        super(ModelNetSabSink, self).__init__()
        sinkhornkeops = SinkhornDistance(eps=eps, max_iter=n_it, cost=distmat2)
        self.enc = nn.Sequential(
            SABSINK(dim_input, dim_hidden, num_heads, ln=ln, sinkhorn=sinkhornkeops),
            SABSINK(dim_hidden, dim_hidden, num_heads, ln=ln, sinkhorn=sinkhornkeops),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMASINK(dim_hidden, num_heads, num_outputs, ln=ln, sinkhorn=sinkhornkeops),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze()


class CorrSabSink(nn.Module):
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_hidden=128,
        num_heads=4,
        ln=False,
        n_it=5,
    ):
        super(CorrSabSink, self).__init__()
        sinkhornkeops = SinkhornDistance(eps=eps, max_iter=n_it, cost=distmat2)
        self.enc = nn.Sequential(
            SABSINK(dim_input, dim_hidden, num_heads, ln=ln, sinkhorn=sinkhornkeops),
            SABSINK(dim_hidden, dim_hidden, num_heads, ln=ln, sinkhorn=sinkhornkeops),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            # PMASINK(dim_hidden, num_heads, num_outputs, ln=ln, sinkhorn=sinkhornkeops),
            SABSINK(dim_hidden, dim_hidden, num_heads, ln=ln, sinkhorn=sinkhornkeops),
            SABSINK(dim_hidden, dim_hidden, num_heads, ln=ln, sinkhorn=sinkhornkeops),
            # nn.Dropout()
        )

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze()