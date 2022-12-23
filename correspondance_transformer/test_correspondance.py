import os
import argparse
import torch
import torch.nn as nn
from DfaustDataset import DfaustActionDataset as Dataset
from models.correformer import CorreFormer
import models.correformer as cf
from models.NNCorr import NNCorr
from models.sinkhorn import SinkhornCorr
import pc_transforms as transforms
import numpy as np

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='sinkhorn', help='nn  | sinkhorn | transformer ')
parser.add_argument('--frames_per_clip', type=int, default=1, help='number of frames in a clip sequence')
parser.add_argument('--batch_size', type=int, default=1, help='number of clips per batch')
parser.add_argument('--n_points', type=int, default=1024, help='number of points in a point cloud')
parser.add_argument('--model_path', type=str, default='./log/dfaust_N1024_ff1024d1024h8_lr0.0001bs32/',
                    help='path to model save dir')
parser.add_argument('--model', type=str, default='001290.pt', help='path to model save dir')
parser.add_argument('--jitter', type=float, default=0.005, help='if larger than 0 : adds random jitter to test points')
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/dfaust/', help='path to dataset')
parser.add_argument('--visualize_results', type=int, default=False, help='visualzies the first subsequence in each batch')
parser.add_argument('--gender', type=str,
                    default='all', help='female | male | all indicating which subset of the dataset to use')
args = parser.parse_args()

test_dataset = Dataset(args.dataset_path,  set='test', n_points=args.n_points, shuffle_points='none',
                       gender=args.gender)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                              pin_memory=True, drop_last=False)


if args.method == 'nn':
    model = NNCorr()
elif args.method == 'sinkhorn':
    model = SinkhornCorr()
elif args.method == 'transformer':
    checkpoints = os.path.join(args.model_path, args.model)
    model = cf.get_correformer(checkpoints)
else:
    raise ValueError("unsupported correspondance model")

model.cuda()
model = nn.DataParallel(model)

correct, total = 0, 0
for test_batchind, data in enumerate(test_dataloader):
    model.train(False)

    points, point_ids = data['points'].cuda(), data['corr_gt'].cuda()
    points2 = torch.roll(points, -1, dims=1).clone().detach()
    points = points[:, 0:-1, :, :].reshape(-1, args.n_points, 3)  # remove first frame pair
    points2 = points2[:, 0:-1, :, :].reshape(-1, args.n_points, 3)  # remove first frame pair

    if args.jitter > 0:
        points2 = transforms.jitter_point_cloud_torch(points2, sigma=args.jitter, clip=0.05)

    for i, frame in enumerate(points):
        out_dict = model(frame.unsqueeze(0), points2[i].unsqueeze(0))
        max_ind = out_dict['corr_idx21']
        true_corr = max_ind == point_ids

        correct += (true_corr).sum().detach().cpu().numpy()
        total += args.n_points

print(args.method+' acc: {}'.format(round(correct/total, 4)))



