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
                                                num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                                batch_first=True, dropout=0.0)

    def single_pass(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x.permute(0, 2, 1)
        out = self.transformer(x, x)
        return out

    def forward(self, x1, x2):
        out1 = self.single_pass(x1)
        out2 = self.single_pass(x2)

        sim_mat = cosine_similarity(out2, out1)
        corr21 = F.softmax(sim_mat, dim=-1)
        # _, max_ind = torch.max(corr, dim=-1)
        with torch.no_grad():
            max_ind21 = torch.argmax(corr21, dim=-1)
            corr12 = F.softmax(sim_mat, dim=-2)
            max_ind12 = torch.argmax(corr12, dim=-2)
        return {'out1': out1, 'out2': out2, 'corr_mat': corr21, 'corr_idx12': max_ind12, 'corr_idx21': max_ind21}


def sort_points(correformer, x):
    b, t, n, k = x.shape
    x = x.cuda()
    sorted_seq = x[:, [0], :, :]
    sorted_frame = x[:, 0, :, :]
    corr_pred = torch.arange(n)[None, None, :].cuda().repeat([b, 1, 1])
    for frame_idx in range(t-1):
        p1 = sorted_frame
        p2 = x[:, frame_idx+1, :, :]
        corre_out_dict = correformer(p1, p2)
        corr_idx12, corr_idx21 = corre_out_dict['corr_idx12'], corre_out_dict['corr_idx21']
        sorted_frame = torch.gather(p2, 1, corr_idx12.unsqueeze(-1).repeat([1, 1, 3]))
        sorted_seq = torch.concat([sorted_seq, sorted_frame.unsqueeze(1)], dim=1)
        corr_pred = torch.concat([corr_pred, corr_idx21.unsqueeze(1)], axis=1)
    return sorted_seq, corr_pred


def get_correformer(correformer_path):
    # load a correformer model from path
    params_str = correformer_path.split("/")[-2].split("_")[2]
    correformer_dims = int(params_str[params_str.index('d') + 1:params_str.index('h')])
    correformer_nheads = int(params_str[params_str.index('h') + 1:])
    correformer = CorreFormer(d_model=correformer_dims, nhead=correformer_nheads, num_encoder_layers=6,
                                   num_decoder_layers=1, dim_feedforward=1024).cuda()
    correformer.load_state_dict(torch.load(correformer_path)["model_state_dict"])
    correformer.eval()
    correformer.train(False)
    return correformer


def compute_corr_loss(gt_corr, corr):
    # compute correspondance loss
    b, n1, n2 = gt_corr.shape
    l1_loss = (gt_corr - corr).square()
    l1_mask = torch.max(gt_corr, 1.0*(torch.rand(b, n1, n2).cuda() < gt_corr.mean()))
    l1_loss = (l1_mask * l1_loss)
    loss = l1_loss.mean()
    return loss