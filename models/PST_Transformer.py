import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from pst_transformer_modules.point_4d_convolution import *
from pst_transformer_modules.transformer_v1 import *

class PSTTransformer(nn.Module):
    def __init__(self, model_cfg, num_class=14, n_frames=32):
        super(PSTTransformer, self).__init__()
        radius = model_cfg['PST_TRANSFORMER']['radius']
        nsamples = model_cfg['PST_TRANSFORMER']['nsamples']
        spatial_stride = model_cfg['PST_TRANSFORMER']['spatial_stride']
        temporal_kernel_size = model_cfg['PST_TRANSFORMER']['temporal_kernel_size']
        temporal_stride = model_cfg['PST_TRANSFORMER']['temporal_stride']
        dim = model_cfg['PST_TRANSFORMER']['dim']
        depth = model_cfg['PST_TRANSFORMER']['depth']
        heads = model_cfg['PST_TRANSFORMER']['heads']
        dim_head = model_cfg['PST_TRANSFORMER']['dim_head']
        dropout1 = model_cfg['PST_TRANSFORMER']['dropout1']
        mlp_dim = model_cfg['PST_TRANSFORMER']['mlp_dim']
        dropout2 = model_cfg['PST_TRANSFORMER']['dropout2']

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                      spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                      temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride,
                                      temporal_padding=[1, 0], operator='+', spatial_pooling='max',
                                      temporal_pooling='max')

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_class),
        )

    def forward(self, input):
        b, t, d, n = input.shape
        input = input.permute(0, 1, 3, 2)  # [B, L, N, 3]
        xyzs, features = self.tube_embedding(input)  # [B, L, n, 3], [B, L, C, n]
        features = features.permute(0, 1, 3, 2)  # [B, L, n, C]

        output = self.transformer(xyzs, features)
        # output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = torch.max(input=output, dim=2, keepdim=False, out=None)[0]

        new_t = int(t / 2)
        output = output.reshape(b * new_t, -1)
        output = self.mlp_head(output)

        output = F.interpolate(output.reshape(b, new_t, -1).permute(0, 2, 1), t, mode='linear', align_corners=True)
        output = F.log_softmax(output, 1)

        return {'pred': output}
