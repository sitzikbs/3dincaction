import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import cosine_similarity

class CorreFormer(nn.Module):
    def __init__(self, d_model=3, nhead=4, num_encoder_layers=4, num_decoder_layers=6, dim_feedforward=256):
        super(CorreFormer, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, d_model, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(d_model)
        self.transformer = torch.nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    def model(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x.permute(0, 2, 1)
        out = self.transformer(x, x)
        return out

    def forward(self, x1, x2):
        out1 = self.model(x1)
        out2 = self.model(x2)

        corr = F.softmax(cosine_similarity(out2, out1), dim=-1)
        _, max_ind = torch.max(corr, dim=-1)
        return {'out1': out1, 'out2': out2, 'corr_mat': corr, 'corr_idx': max_ind}