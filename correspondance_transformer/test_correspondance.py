import os
import argparse
import torch
import torch.nn as nn
from DfaustDataset import DfaustActionDataset as Dataset
from DfaustDataset import DfaustActionClipsDataset
import models.correformer as cf
from models.NNCorr import NNCorr
from models.sinkhorn import SinkhornCorr
import pc_transforms as transforms
import numpy as np
import visualization as vis
import importlib
import csv
import pandas as pd

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='sinkhorn', help='nn  | sinkhorn | transformer ')
parser.add_argument('--frames_per_clip', type=int, default=1, help='number of frames in a clip sequence')
parser.add_argument('--batch_size', type=int, default=1, help='number of clips per batch')
parser.add_argument('--n_points', type=int, default=1024, help='number of points in a point cloud')
parser.add_argument('--model_path', type=str, default='./log/dfaust_N1024ff1024_d1024h8_ttypenonelr0.0001bs32reg_cat_ce2/',
                    help='path to model save dir')
parser.add_argument('--model', type=str, default='000250.pt', help='path to model save dir')
parser.add_argument('--jitter', type=float, default=0.01, help='if larger than 0 : adds random jitter to test points')
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/dfaust/', help='path to dataset')
parser.add_argument('--visualize_results', type=int, default=False, help='visualzies the first subsequence in each batch')
parser.add_argument('--gender', type=str,
                    default='all', help='female | male | all indicating which subset of the dataset to use')
parser.add_argument('--sinkhorn_n_iters', type=int, default=100, help='number of maximum iterations for sinkhorn')
args = parser.parse_args()

test_dataset = Dataset(args.dataset_path,  set='test', n_points=args.n_points, shuffle_points='fps_each',
                       gender=args.gender)
# test_dataset = DfaustActionClipsDataset(args.dataset_path, frames_per_clip=2, set='test', n_points=args.n_points,
#                         shuffle_points='each', data_augmentation=['jitter'], gender=args.gender)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                              pin_memory=True, drop_last=False)


# set up the output directory and output file once if they do not exis
out_path = './results/'
if not os.path.exists((out_path)):
    os.makedirs((out_path))
    with open(os.path.join(out_path, 'test_correspondence_results.csv'), 'a', encoding='UTF8') as f:
        header = ['method', 'jitter', 'acc']
        writer = csv.writer(f)
        writer.writerow(header)


if args.method == 'nn':
    model = NNCorr(dist_type='euclid')
elif args.method == 'sinkhorn':
    model = SinkhornCorr(args.sinkhorn_n_iters)
elif args.method == 'transformer':
    #TODO load correformer.py file from model_path directory using importlib
    checkpoints = os.path.join(args.model_path, args.model)
    model = cf.get_correformer(checkpoints)
else:
    raise ValueError("unsupported correspondance model")

model.cuda()
model = nn.DataParallel(model)

correct, total = 0, 0
for test_batchind, data in enumerate(test_dataloader):
    model.train(False)
    model.eval()


    points, point_ids = data['points'].cuda(), data['corr_gt'].cuda()
    points2 = torch.roll(points, -1, dims=1).detach().clone()
    points = points[:, 0:-1, :, :].reshape(-1, args.n_points, 3)  # remove first frame pair
    points2 = points2[:, 0:-1, :, :].reshape(-1, args.n_points, 3)  # remove first frame pair

    if args.jitter > 0.0:
        points = transforms.jitter_point_cloud_torch(points, sigma=args.jitter, clip=0.05)
        points2 = transforms.jitter_point_cloud_torch(points2, sigma=args.jitter, clip=0.05)

    for i, frame in enumerate(points):
        point_ids = torch.randperm(args.n_points).cuda()
        target_points = points2[i, point_ids, :].unsqueeze(0)
        out_dict = model(frame.unsqueeze(0), target_points)
        # out_dict = model(frame.unsqueeze(0), points2[i].unsqueeze(0))
        max_ind = out_dict['corr_idx21']
        true_corr = max_ind == point_ids
        correct += (true_corr).sum().detach().cpu().numpy()
        total += args.n_points

print(args.method+' acc: {}'.format(round(correct/total, 4)))


# write result to csv file
results_filename = os.path.join(out_path, 'test_correspondence_results.csv')
with open(results_filename, 'a', encoding='UTF8') as f:
    if args.method == 'transformer':
        method = args.method + '_' + checkpoints
    elif args.method == 'sinkhorn':
        method = args.method + '_'+ args.sinkhorn_n_iters
    else:
        method = args.method
    data = [method, args.jitter, round(correct / total, 4)]
    writer = csv.writer(f)
    writer.writerow(data)     # write the data



