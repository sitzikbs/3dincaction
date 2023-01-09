# minimum working example to debug the memory leak
from typing import Union
from scipy.spatial import KDTree
import torch
import tqdm
import numpy as np
import torch.nn as nn
import geomloss


def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(-2)
    y_lin = y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    return C

class sinkhorn(nn.Module):
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
    def __init__(self, eps=1, max_iter=1, cost=cost_matrix, thresh=1e-2):
        super(sinkhorn, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.cost = cost
        self.thresh = thresh

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

        # Sinkhorn iterations
        for i in range(self.max_iter):
            # u_prev = u
            # v_prev = v
            # summand_u = (-C + v) / self.eps
            # u = self.eps * (torch.log(mu).squeeze() - summand_u.logsumexp(dim=-1).squeeze())
            #
            # summand_v = (-C + u) / self.eps
            # v = self.eps * (torch.log(nu).squeeze() - summand_v.logsumexp(dim=-2).squeeze())
            #
            # max_err_u = torch.max(torch.abs(u_prev - u))
            # max_err_v = torch.max(torch.abs(v_prev - v))
            #
            # if max_err_u < self.thresh and max_err_v < self.thresh:
            #     break
            if i % 2 == 0:
                u1 = u  # useful to check the update
                u = self.eps * (torch.log(mu) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
                err = (u - u1).abs().sum(-1).mean()
            else:
                v = self.eps * (torch.log(nu) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            if err.item() < thresh:
                break


        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        approx_corr_1 = pi.argmax(dim=-1).squeeze(-1)
        approx_corr_2 = pi.argmax(dim=-2).squeeze(-1)

        # Sinkhorn distance
        if u.shape[0] > v.shape[0]:
            distance = (pi * C).sum(dim=-1).sum()
        else:
            distance = (pi * C).sum(dim=-2).sum()
        return distance, approx_corr_1, approx_corr_2

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps



    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

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
    batch_size = 1
    n_points = 1024
    n_epochs = 10000

    model = sinkhorn()
    dataset = NoiseGenerator(n_points, n_samples=10000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                  pin_memory=True)
    for epoch in range(n_epochs):
        for batch_ind, data in enumerate(dataloader):
            print("Epoch {}, Batch {}".format(epoch, batch_ind))
            points1 = data['points']
            points2 = local_distort(points1)
            with torch.no_grad():
                output = model(points1.squeeze().cuda(), points2.squeeze().cuda())