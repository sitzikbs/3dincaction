import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from p4transformer_modules.point_4d_convolution import *
from p4transformer_modules.transformer import *


class P4Transformer(nn.Module):
    def __init__(self, model_cfg, num_class=14, n_frames=32):
        super().__init__()
        radius = model_cfg['P4TRANSFORMER']['radius']
        nsamples = model_cfg['P4TRANSFORMER']['nsamples']
        spatial_stride = model_cfg['P4TRANSFORMER']['spatial_stride']
        temporal_kernel_size = model_cfg['P4TRANSFORMER']['temporal_kernel_size']
        temporal_stride = model_cfg['P4TRANSFORMER']['temporal_stride']
        dim = model_cfg['P4TRANSFORMER']['dim']
        depth = model_cfg['P4TRANSFORMER']['depth']
        heads = model_cfg['P4TRANSFORMER']['heads']
        dim_head = model_cfg['P4TRANSFORMER']['dim_head']
        dropout1 = model_cfg['P4TRANSFORMER']['dropout1']
        mlp_dim = model_cfg['P4TRANSFORMER']['mlp_dim']
        dropout2 = model_cfg['P4TRANSFORMER']['dropout2']
        emb_relu = model_cfg['P4TRANSFORMER']['emb_relu']

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_class),
        )

    def forward(self, input):          # [B, L, N, 3]
        B, T, d, N = input.shape
        input = input.permute(0, 1, 3, 2)  # [B, L, N, 3]
        xyzs, features = self.tube_embedding(input)         # [B, L, n, 3], [B, L, C, n]

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=input.device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))      # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)         # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.transformer(embedding)

        # output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        # output = self.mlp_head(output)
        # return output

        # TODO: add support for per frame prediction
        new_t = len(xyzs)
        output = output.reshape(B, new_t, -1, output.shape[-1])  # [B, L, n, C]
        output = torch.max(output, dim=2, keepdim=False, out=None)[0]  # [B, L, C]
        output = self.mlp_head(output.reshape(B * new_t, -1))

        output = F.interpolate(output.reshape(B, new_t, -1).permute(0, 2, 1), T, mode='linear', align_corners=True)
        output = F.log_softmax(output, 1)

        return {'pred': output}



