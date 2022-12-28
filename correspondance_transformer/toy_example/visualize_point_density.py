import os
import argparse
import utils
import torch
from DfaustDataset import DfaustActionDataset as Dataset
import visualization
import numpy as np
from scipy.spatial import cKDTree

def compute_local_mean_dist(points, k=16):
    tree = cKDTree(points[0])
    d, inds = tree.query(points[0], k)
    mean_d = np.mean(d, -1)
    return mean_d

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/dfaust/', help='path to dataset')
parser.add_argument('--set', type=str, default='test', help='test | train set to evaluate ')
args = parser.parse_args()

test_dataset = Dataset(args.dataset_path,  set=args.set, n_points=1024, shuffle_points='each')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                              pin_memory=True)

# Iterate over data.
for test_batchind, data in enumerate(test_dataloader):
    inputs, labels_int, seq_idx = data['points'], data['labels'], data['seq_idx']
    inputs = inputs.permute(0, 1, 3, 2).cuda().requires_grad_().contiguous().squeeze()
    points = inputs[0].unsqueeze(0).permute(0, 2, 1).detach().cpu().numpy()
    mean_d = compute_local_mean_dist(points, k=16)
    visualization.plot_pc_pv(points, color=mean_d[None, :], cmap='jet')

