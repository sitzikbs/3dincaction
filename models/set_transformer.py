import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

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
    def __init__(self, model_cfg, num_classes=40):
        super(SetTransformer, self).__init__()
        cfg = model_cfg['SET_TRANSFORMER']
        self.num_classes = num_classes
        dim_input, num_outputs = cfg['dim_input'], cfg['num_outputs']
        num_inds, dim_hidden, num_heads, ln = cfg['num_inds'], cfg['dim_hidden'], cfg['num_heads'], cfg['ln']
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            # nn.Linear(dim_hidden, num_classes),
        )

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze()


class SetTransformerTemporal(SetTransformer):
    def __init__(self, model_cfg, num_class=40, n_frames=32):
        SetTransformer.__init__(self, model_cfg, num_classes=num_class)
        self.temporal_smoothing = model_cfg['SET_TRANSFORMER']['temporal_smoothing']
        self.dim_hidden = model_cfg['SET_TRANSFORMER']['dim_hidden']
        self.classifier12 = nn.Sequential(
            nn.Linear(self.dim_hidden, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        if not self.temporal_smoothing == 0:
            self.temporalconv = torch.nn.Conv1d(256, 256, self.temporal_smoothing, 1, padding='same')
            self.bnt = nn.BatchNorm1d(256)

        self.final_layer = nn.Linear(256, num_class)

    def forward(self, X):
        b, t, d, n = X.shape
        X = X.reshape(b*t, d, n).permute(0, 2, 1)
        out = self.dec(self.enc(X)).squeeze()
        # out = out.reshape(b, t, self.dim_hidden).permute(0, 2, 1)
        out = self.classifier12(out)
        if not self.temporal_smoothing == 0:
            out = F.relu(self.bnt(self.temporalconv(out.reshape(b, t, -1).permute(0, 2, 1))))
            out = out.permute(0, 2, 1).reshape(b*t, -1)
        out = self.final_layer(out)
        out = F.log_softmax(out, -1).reshape(b, t, -1).permute(0, 2, 1)
        return {'pred': out}


class SetTransformerCross(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformerCross, self).__init__()
        self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln),
                # SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        )
        self.dec = nn.Sequential(
                # PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                # SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))
