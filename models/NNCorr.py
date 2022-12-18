import torch
import torch.nn as nn


class NNCorr(nn.Module):
    def __init__(self, ):
        super(NNCorr, self).__init__()

    def forward(self, x1, x2):
        corr_mat = torch.cdist(x1, x2, p=2.0)
        max_ind21 = torch.argmin(corr_mat, dim=-1)
        max_ind12 = torch.argmin(corr_mat, dim=-2)
        return {'out1': x1, 'out2': x2, 'corr_mat': corr_mat, 'corr_idx12': max_ind12, 'corr_idx21': max_ind21}






