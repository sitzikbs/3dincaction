import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F

import models.correformer as cf
from DfaustDataset import DfaustActionDataset as Dataset
import pc_transforms as transforms
import visualization as vis
from models.NNCorr import NNCorr
import matplotlib.pyplot as plt

title_dict = {'sim_mat': 'Similarity matrix', 'corr21': 'softmax cols', 'corr12': 'softmax rows',
              'best_buddies_mat': 'best buddies', 'max_corr21_mat': 'max_corr21', 'max_corr12_mat': 'max_corr12',
              'err_mat': 'Error mat', 'GT': 'GT', '3Dpc_diff': 'diff overlay'}
def get_result_matrices(out_dict, points2, point_ids):
    max_ind21, max_ind12 = out_dict['corr_idx21'], out_dict['corr_idx12']
    sim_mat = out_dict['sim_mat'].squeeze().detach().cpu().numpy()
    corr21 = F.softmax(out_dict['sim_mat'], dim=-1).squeeze().detach().cpu().numpy()
    corr12 = F.softmax(out_dict['sim_mat'], dim=-2).squeeze().detach().cpu().numpy()
    buddies_mat = out_dict['sim_mat'] > 0.6
    best_buddies_mat = (buddies_mat * buddies_mat.permute(0, 2, 1)).squeeze().detach().cpu().numpy()
    max_corr21_mat = (max_ind21[0].unsqueeze(-1) == torch.arange(args.n_points).cuda()).float().cpu().numpy()
    max_corr12_mat = (max_ind12[0].unsqueeze(-1) == torch.arange(args.n_points).cuda()).float().cpu().numpy()
    err_mat = np.abs(gt_corr - max_corr21_mat)
    pc_diff_pv = vis.get_pc_pv_image(points2.detach().cpu().numpy(), text=None,
                                     color=torch.logical_not(max_ind21[0] == point_ids).cpu().detach().numpy(),
                                     point_size=25, cmap='jet')#.transpose(2, 0, 1)

    result_mat_dict = {'sim_mat': sim_mat, 'corr21': corr21, 'corr12': corr12, 'best_buddies_mat': best_buddies_mat,
                       'max_corr21_mat': max_corr21_mat, 'max_corr12_mat': max_corr12_mat, 'err_mat': err_mat,
                       '3Dpc_diff': pc_diff_pv}

    return result_mat_dict
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/dfaust/', help='path to dataset')
parser.add_argument('--set', type=str, default='test', help='test | train set to evaluate ')
parser.add_argument('--model_path', type=str,
                    default='../log/jitter/dfaust_N1024ff1024_d1024h8_ttypenonelr0.0001bs8debug_bbl/',
                    help='path to model save dir')
parser.add_argument('--model', type=str, default='000050.pt', help='path to model save dir')
parser.add_argument('--jitter', type=float, default=0.005,
                    help='if larger than 0 : adds random jitter to test points')
parser.add_argument('--n_points', type=int, default=1024, help='number of points in each model')
args = parser.parse_args()

test_dataset = Dataset(args.dataset_path,  set=args.set, n_points=args.n_points, shuffle_points='each')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                              pin_memory=True)

checkpoints = os.path.join(args.model_path, args.model)
model = cf.get_correformer(checkpoints)
NN_model = NNCorr()
correct = 0
total = 0
# Iterate over data.
for test_batchind, data in enumerate(test_dataloader):
    model.train(False)
    model.eval()
    points, point_ids = data['points'].cuda(), data['corr_gt'].cuda()
    points2 = torch.roll(points, -1, dims=1).detach().clone()
    points = points[:, 0:-1, :, :].reshape(-1, args.n_points, 3)  # remove first frame pair
    points2 = points2[:, 0:-1, :, :].reshape(-1, args.n_points, 3)  # remove first frame pair

    if args.jitter > 0.0:
        points = transforms.jitter_point_cloud_torch(points, sigma=args.jitter, clip=5 * args.jitter)
        points2 = transforms.jitter_point_cloud_torch(points2, sigma=args.jitter, clip=5 * args.jitter)

    for i, frame in enumerate(points):
        point_ids = torch.randperm(args.n_points).cuda()
        target_points = points2[i, point_ids, :].unsqueeze(0)
        out_dict = model(frame.unsqueeze(0), target_points)

        max_ind21, max_ind12 = out_dict['corr_idx21'], out_dict['corr_idx12']
        gt_corr = (point_ids.unsqueeze(-1) == torch.arange(args.n_points).cuda()).float().cpu().numpy()

        # Compute matrices for model
        mat_dict_corr = get_result_matrices(out_dict, target_points.squeeze(), point_ids)

        # # Compute matrices for NN
        # nn_out_dict = NN_model(frame.unsqueeze(0), target_points)
        # sim_mat = out_dict['sim_mat'].squeeze().detach().cpu().numpy()

        # print and visualize
        true_corr = max_ind21 == point_ids
        correct += (true_corr).sum().detach().cpu().numpy()
        total += args.n_points
        instance_acc = correct / total
        print("Pair correspondance accuracy: {}".format(instance_acc))

        vis.plot_correformer_outputs(mat_dict_corr, title_dict, title_text='Correformer results')
        # vis.pc_seq_vis(target_points.squeeze().cpu().detach().numpy())
        # diff_colors = torch.logical_not(max_ind21[0] == point_ids).unsqueeze(0).cpu().detach().numpy()
        # vis.plot_pc_pv(target_points.cpu().detach().numpy(), text=None, color=diff_colors, cmap='jet', point_size=25)
        plt.close()
    acc = correct/total