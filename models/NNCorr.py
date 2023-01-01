import torch
import torch.nn as nn
from utils import cosine_similarity

class NNCorr(nn.Module):
    def __init__(self, dist_type='euclid'):
        super(NNCorr, self).__init__()
        self.dist_type = dist_type

    def forward(self, x1, x2):
        if self.dist_type == 'euclid':
            corr_mat = torch.cdist(x2, x1, p=2.0)
            max_ind21 = torch.argmin(corr_mat, dim=-1)
            max_ind12 = torch.argmin(corr_mat, dim=-2)
        elif self.dist_type == 'cos':
            corr_mat = cosine_similarity(x2, x1)
            max_ind21 = torch.argmax(corr_mat, dim=-1)
            max_ind12 = torch.argmax(corr_mat, dim=-2)

        return {'out1': x1, 'out2': x2, 'corr_mat': corr_mat, 'corr_idx12': max_ind12, 'corr_idx21': max_ind21}






