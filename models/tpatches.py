import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
import time
from models.extractors import PatchletsExtractor, PatchletsExtractorBidirectional, PatchletsExtractorStrided



class PatchletTemporalConv(nn.Module):
    def __init__(self, in_channel, temporal_conv, mlp, use_attn=False, attn_num_heads=4, temporal_stride=1):
        super(PatchletTemporalConv, self).__init__()
        self.use_attn = use_attn

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.temporal_stride = temporal_stride
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv3d(last_channel, out_channel, 1))
            # self.mlp_convs.append(nn.Conv3d(last_channel, out_channel, [1, temporal_conv, k], 1, padding='same'))
            self.mlp_bns.append(nn.BatchNorm3d(out_channel))
            last_channel = out_channel

        if self.use_attn:
            self.multihead_attn = nn.MultiheadAttention(last_channel, attn_num_heads, batch_first=True)
        else:
            if self.temporal_stride == 1:
                self.temporal_conv = nn.Conv2d(out_channel, out_channel, [1, temporal_conv], padding='same')
            else:
                self.temporal_conv = nn.Conv2d(out_channel, out_channel, [1, temporal_conv], [1, temporal_stride])
            self.bnt = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        b, d, n, t, k = x.shape
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x = F.relu(bn(conv(x)))

        x = torch.max(x, -1)[0] # pool neighbors to get patch representation
        # x = torch.mean(x, -1)  # pool neighbors to get patch representation

        if self.use_attn:
            x = x.permute(0, 2, 3, 1)
            b, n, f, c = x.shape
            x = x.reshape(b * n, f, c)
            attn, _ = self.multihead_attn(x, x, x)
            x = attn.view(b, n, f, c).permute(0, 2, 1, 3)
        else:
            x = F.relu(self.bnt(self.temporal_conv(x)))  # convolve temporally to improve patch representation
            if not self.temporal_stride == 1:
                x = x.repeat(1, 1, 1, int(t/x.shape[-1]))
            x = x.permute(0, 3, 2, 1)
        return x


class PointMLP(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointMLP, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, x):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x = F.relu(bn(conv(x)))

        return x.permute(0, 2, 3, 1)


class TPatches(nn.Module):
    def __init__(self, model_cfg, num_class, n_frames=32, timeit=False): # TODO: remove
        super(TPatches, self).__init__()
        self.timeit = timeit
        in_channel = model_cfg.get('in_channel', 3)
        cfg = model_cfg['PATCHLET']
        self.k_list = cfg.get('k', [16, 16, 16])
        self.sample_mode = cfg['sample_mode']
        self.centroid_jitter = cfg['centroid_jitter']
        self.n_frames = n_frames
        self.downsample_method = cfg['downsample_method']
        self.radius = cfg['radius']
        self.type = cfg.get('type', 'origin')
        attn_num_heads = cfg.get('attn_num_heads')
        self.extractor_type = cfg.get('extractor_type', 'bidirectional')
        self.temporal_stride = cfg.get('temporal_stride', 1)
        self.local_temp_convs = cfg.get('local_temp_convs', [8, 8, 8])
        npoints = cfg.get('npoints', [512, 128, None])
        temp_conv = cfg.get('temp_conv', n_frames)

        if self.extractor_type == 'bidirectional':
            Extractor = PatchletsExtractorBidirectional
        elif self.extractor_type == 'strided':
            Extractor = PatchletsExtractorStrided
        elif self.extractor_type == 'vanilla':
            Extractor = PatchletsExtractor
        else:
            raise ValueError("Unsupported extractor type")

        use_attn = False
        if self.type == 'attn_all_layers':
            use_attn = True

        # self.point_mlp = PointMLP(in_channel=in_channel, mlp=[64, 64, 128])
        self.patchlet_extractor1 = Extractor(k=self.k_list[0], sample_mode=self.sample_mode, npoints=npoints[0],
                                             add_centroid_jitter=self.centroid_jitter,
                                             downsample_method=self.downsample_method, radius=self.radius[0],
                                             temporal_stride=self.temporal_stride)
        self.patchlet_temporal_conv1 = PatchletTemporalConv(in_channel=in_channel,
                                                            temporal_conv=self.local_temp_convs[0], mlp=[64, 64, 128],
                                                            use_attn=use_attn, attn_num_heads=attn_num_heads,
                                                            temporal_stride=self.temporal_stride)
        self.patchlet_extractor2 = Extractor(k=self.k_list[1], sample_mode=self.sample_mode, npoints=npoints[1],
                                             add_centroid_jitter=self.centroid_jitter,
                                             downsample_method=self.downsample_method, radius=self.radius[1],
                                             temporal_stride=self.temporal_stride)
        self.patchlet_temporal_conv2 = PatchletTemporalConv(in_channel=128+3, temporal_conv=self.local_temp_convs[1],
                                                            mlp=[128, 128, 256],
                                                            use_attn=use_attn, attn_num_heads=attn_num_heads,
                                                            temporal_stride=self.temporal_stride)
        self.patchlet_extractor3 = Extractor(k=self.k_list[2], sample_mode=self.sample_mode, npoints=npoints[2],
                                             add_centroid_jitter=self.centroid_jitter,
                                             downsample_method=None, radius=self.radius[2],
                                             temporal_stride=self.temporal_stride)
        self.patchlet_temporal_conv3 = PatchletTemporalConv(in_channel=256+3, temporal_conv=self.local_temp_convs[2],
                                                            mlp=[256, 512, 1024],
                                                            use_attn=use_attn, attn_num_heads=attn_num_heads,
                                                            temporal_stride=self.temporal_stride)

        # self.temporal_pool = torch.nn.MaxPool3d([n_frames, 1, 1])
        # self.temporal_pool = torch.nn.AvgPool2d(3, stride=1, padding=1)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

        # self.bnt = nn.BatchNorm1d(1024)
        # self.temporalconv1 = torch.nn.Conv1d(1024, 1024, int(n_frames/4), 1, padding='same')
        self.temporalconv2 = torch.nn.Conv1d(256, 256, temp_conv, 1, padding='same')
        self.bn3 = nn.BatchNorm1d(256)

        if self.type == 'attn_last_layer' or self.type == 'attn_all_layers':
            embed_dim = 256
            self.multihead_attn = nn.MultiheadAttention(embed_dim, attn_num_heads, batch_first=True)

    def forward(self, xyz):
        b, t, d, n = xyz.shape
        xyz = xyz.permute(0, 1, 3, 2)

        if d > 3:
            patchlet_feats = xyz[:, :, :, 3:]
            xyz = xyz[:, :, :, :3]
        else:
            patchlet_feats = None

        if self.timeit:
            t_start_ext_1 = time.time()
        patchlet_dict = self.patchlet_extractor1(xyz, patchlet_feats)
        if self.timeit:
            t_end_ext_1 = time.time()
            t_ext_1 = t_end_ext_1 - t_start_ext_1

        xyz0 = patchlet_dict['patchlet_points']
        if patchlet_feats is not None:
            patchlet_feats = patchlet_dict['patchlet_feats'].permute(0, 4, 2, 1, 3)
        else:
            patchlet_feats = patchlet_dict['normalized_patchlet_points'].permute(0, 4, 2, 1, 3)

        if self.timeit:
            t_start_tempconv_1 = time.time()
        patchlet_feats = self.patchlet_temporal_conv1(patchlet_feats)  # [b, d+k, npoint, t, nsample]
        if self.timeit:
            t_end_tempconv_1 = time.time()
            t_tempconv_1 = t_end_tempconv_1 - t_start_tempconv_1

        if self.timeit:
            t_start_ext_2 = time.time()
        patchlet_dict = self.patchlet_extractor2(xyz0[:, :, :, 0, :], patchlet_feats)
        if self.timeit:
            t_end_ext_2 = time.time()
            t_ext_2 = t_end_ext_2 - t_start_ext_2

        xyz = patchlet_dict['patchlet_points']
        patchlet_feats = patchlet_dict['patchlet_feats'].permute(0, 4, 2, 1, 3)

        if self.timeit:
            t_start_tempconv_2 = time.time()
        patchlet_feats = self.patchlet_temporal_conv2(patchlet_feats)  # [b, d+k, npoint, t, nsample]
        if self.timeit:
            t_end_tempconv_2 = time.time()
            t_tempconv_2 = t_end_tempconv_2 - t_start_tempconv_2

        if self.timeit:
            t_start_ext_3 = time.time()
        patchlet_dict = self.patchlet_extractor3(xyz[:, :, :, 0, :], patchlet_feats)
        if self.timeit:
            t_end_ext_3 = time.time()
            t_ext_3 = t_end_ext_3 - t_start_ext_3

        xyz = patchlet_dict['patchlet_points']
        patchlet_feats = patchlet_dict['patchlet_feats'].permute(0, 4, 2, 1, 3)

        if self.timeit:
            t_start_tempconv_3 = time.time()
        patchlet_feats = self.patchlet_temporal_conv3(patchlet_feats)  # [b, d+k, npoint, t, nsample]
        if self.timeit:
            t_end_tempconv_3 = time.time()
            t_tempconv_3 = t_end_tempconv_3 - t_start_tempconv_3

        if self.timeit:
            t_start_cls = time.time()

        xyz, patchlet_feats = xyz.squeeze(-1), patchlet_feats.squeeze(-1)
        x = torch.max(patchlet_feats, -2)[0]
        # x = torch.mean(patchlet_feats, -2)
        x = x.reshape(b*t, 1024)

        x = self.drop1(F.relu(self.bn1(self.fc1(x).reshape(b, t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512))
        x = self.drop2(F.relu(self.bn2(self.fc2(x).reshape(b, t, 256).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 256))

        # learn a temporal filter on all per-frame global representations
        if self.type == 'origin':
            x = F.relu(
                self.bn3(self.temporalconv2(x.reshape(b, t, 256).permute(0, 2, 1)).permute(0, 2, 1).reshape(-1, 256)))
        elif self.type == 'skip_con':
            out_temp_conv = F.relu(self.bn3(self.temporalconv2(x.reshape(b, t, 256).permute(0, 2, 1))))
            if not self.temporal_stride == 1:
                out_temp_conv = F.interpolate(out_temp_conv, t, mode='linear', align_corners=True)
            x = x + out_temp_conv.reshape(-1, 256)
        elif self.type == 'attn_last_layer' or self.type == 'attn_all_layers':
            x = x.view(b, t, -1)
            attn_output, _ = self.multihead_attn(x, x, x)
            x = attn_output.reshape(b * t, -1)
        else:
            raise NotImplementedError

        x = self.fc3(x)

        # x = self.temporal_pool(x.reshape(b, t, -1))
        x = F.log_softmax(x, -1)

        if self.timeit:
            t_end_cls = time.time()
            t_classifier = t_end_cls - t_start_cls

            time_dict = {
                'patchlet_extractor': [t_ext_1, t_ext_2, t_ext_3],
                'patchlet_temporal_conv': [t_tempconv_1, t_tempconv_2, t_tempconv_3],
                'patchlet_classifier': [t_classifier],
            }
        else:
            time_dict = {}

        return {'pred': x.reshape(b, t, -1).permute([0, 2, 1]), 'features': patchlet_feats,
                'patchlet_points': xyz0, 'time_dict': time_dict}
