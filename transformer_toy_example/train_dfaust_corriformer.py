# from modules import ISAB, PMA, SAB
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import argparse
# from data_spheres import SphereGenerator
from DfaustDataset import DfaustActionClipsDataset as Dataset
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append('../')
import visualization as vis
from models.correformer import CorreFormer
from utils import cosine_similarity


def log_images(writer,
               log_dict, iter):
    for key in log_dict.keys():
        writer.add_image(key, log_dict[key], iter)


def log_scalars(writer, log_dict, iter):
    for key in log_dict.keys():
        writer.add_scalar(key, log_dict[key], iter)


parser = argparse.ArgumentParser()
parser.add_argument("--n_points", type=int, default=1024)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--dim", type=int, default=1024)
parser.add_argument("--n_heads", type=int, default=16)
parser.add_argument("--train_epochs", type=int, default=500000)
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/dfaust/', help='path to dataset')
parser.add_argument('--frames_per_clip', type=int, default=1, help='number of frames in a clip sequence')
point_size = 25
args = parser.parse_args()
args.exp_name = f"dfaust_N{args.n_points}_d{args.dim}h{args.n_heads}_lr{args.learning_rate}bs{args.batch_size}"
log_dir = "./log/" + args.exp_name
model_path = log_dir + "/model"
writer = SummaryWriter(os.path.join(log_dir, 'train'))
test_writer = SummaryWriter(os.path.join(log_dir, 'test'))

# Set up data
train_dataset = Dataset(args.dataset_path, frames_per_clip=args.frames_per_clip + 1, set='train', n_points=args.n_points,
                        shuffle_points='each', data_augmentation=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                                               pin_memory=True, shuffle=True, drop_last=True)
test_dataset = Dataset(args.dataset_path, frames_per_clip=args.frames_per_clip + 1, set='test', n_points=args.n_points,
                       shuffle_points='each')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                              pin_memory=True)
test_enum = enumerate(test_dataloader, 0)

# set up model
model = CorreFormer(d_model=args.dim, nhead=args.n_heads, num_encoder_layers=6, num_decoder_layers=1, dim_feedforward=1024)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
model = nn.DataParallel(model).cuda()
criterion = torch.nn.MSELoss(reduction='none')

for epoch in range(args.train_epochs):
    model.train()
    losses, total, correct = [], 0, 0
    for batch_idx, data in enumerate(train_dataloader):

        points = data['points'].cuda()
        points2 = torch.roll(points, -1, dims=1).detach().clone()
        points = points[:, 0:-1, :, :].reshape(-1, args.n_points, 3) # remove first frame pair
        points2 = points2[:, 0:-1, :, :].reshape(-1, args.n_points, 3) # remove first frame pair
        point_ids = torch.randperm(args.n_points).cuda()
        points2 = points2[:, point_ids, :]
        gt_corr = (point_ids.unsqueeze(-1) == torch.arange(args.n_points).cuda().unsqueeze(-2)).float().unsqueeze(0).repeat([args.batch_size, 1, 1]).cuda()

        out_dict = model(points, points2)
        out1, out2, corr, max_ind = out_dict['out1'], out_dict['out2'], out_dict['corr_mat'], out_dict['corr_idx']

        l1_loss = criterion(gt_corr, corr)
        l1_mask = torch.max(gt_corr, 1.0*(torch.rand(args.batch_size, args.n_points, args.n_points).cuda() < gt_corr.mean()))
        l1_loss = (l1_mask * l1_loss).mean()

        loss = l1_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += args.n_points*args.batch_size
        correct += (max_ind == point_ids).sum().detach().cpu().numpy()
        avg_acc = correct / total
        iter = epoch * len(train_dataset) + batch_idx

        loss_log_dict = {"acc": avg_acc, "losses/l1_loss": l1_loss.detach().cpu().numpy(),
                         "losses/total_loss": loss.detach().cpu().numpy()}
        log_scalars(writer, loss_log_dict, iter)

        print(f"Epoch {epoch} batch {batch_idx}: train loss {loss:.3f}")

    if epoch % 5 == 0:

        max_corr = (max_ind[0].unsqueeze(-1) == torch.arange(args.n_points).cuda().unsqueeze(-2)).float().unsqueeze(0).repeat([args.batch_size, 1, 1]).cuda()
        pc1_pv = vis.get_pc_pv_image(points[0].detach().cpu().numpy(), text=None,
                                     color=np.arange(args.n_points), point_size=point_size).transpose(2, 0, 1)
        pc2_pv = vis.get_pc_pv_image(points2[0].detach().cpu().numpy(),  text=None,
                                     color=max_ind[0].detach().cpu().numpy(), point_size=point_size).transpose(2, 0, 1)
        image_log_dict = {"images/GT": gt_corr[0].unsqueeze(0).detach().cpu().numpy(),
                          "images/corr": corr[0].unsqueeze(0).detach().cpu().numpy(),
                          "images/diff": torch.abs(gt_corr[0] - corr[0]).unsqueeze(0).detach().cpu().numpy(),
                          "images/max": max_corr[0].unsqueeze(0).detach().cpu().numpy(),
                          "images/max_diff": torch.abs(gt_corr[0] - max_corr[0]).unsqueeze(0).detach().cpu().numpy(),
                          "3D_corr_images/source": pc1_pv, "3D_corr_images/target": pc2_pv}
        log_images(writer, image_log_dict, epoch)
        print(f"Epoch {epoch}: train loss {loss:.3f}")
        writer.flush()

        ################################################### run test ###########################################
        with torch.no_grad():
            model.eval()  # Set model to evaluate mode
            test_batchind, data = next(test_enum)
            if test_batchind == len(test_dataloader):
                test_enum = enumerate(test_dataloader, 0)
            points = data['points']
            points2 = torch.roll(points, -1, dims=1).detach().clone()
            points = points[:, 0:-1, :, :].reshape(-1, args.n_points, 3) # remove first frame pair
            points2 = points2[:, 0:-1, :, :].reshape(-1, args.n_points, 3) # remove first frame pair
            point_ids = torch.randperm(args.n_points).cuda()
            points2 = points2[:, point_ids, :]
            gt_corr = (point_ids.unsqueeze(-1) == torch.arange(args.n_points).cuda().unsqueeze(-2)).float().unsqueeze(0).repeat([args.batch_size, 1, 1]).cuda()

            out_dict = model(points, points2)
            out1, out2, corr, max_ind = out_dict['out1'], out_dict['out2'], out_dict['corr_mat'], out_dict['corr_idx']

            l1_loss = criterion(gt_corr, corr)
            l1_mask = torch.max(gt_corr, 1.0*(torch.rand(args.batch_size, args.n_points, args.n_points).cuda() < gt_corr.mean()))
            l1_loss = (l1_mask * l1_loss).mean()

            loss = l1_loss


            test_correct = (max_ind == point_ids).sum().detach().cpu().numpy()
            avg_acc = test_correct / (args.n_points*args.batch_size)
            loss_log_dict = {"acc": avg_acc, "losses/l1_loss": l1_loss.detach().cpu().numpy(),
                        "losses/total_loss": loss.detach().cpu().numpy()}
            log_scalars(test_writer, loss_log_dict, iter)
            max_corr = (max_ind[0].unsqueeze(-1) == torch.arange(args.n_points).cuda().unsqueeze(-2)).float().unsqueeze(
                0).repeat([args.batch_size, 1, 1]).cuda()
            pc1_pv = vis.get_pc_pv_image(points[0].detach().cpu().numpy(), text=None, color=np.arange(args.n_points),
                                         point_size=point_size).transpose(2, 0, 1)
            pc2_pv = vis.get_pc_pv_image(points2[0].detach().cpu().numpy(), text=None,
                                         color=max_ind[0].detach().cpu().numpy(), point_size=point_size).transpose(2, 0, 1)
            image_log_dict = {"images/GT": gt_corr[0].unsqueeze(0).detach().cpu().numpy(),
                              "images/corr": corr[0].unsqueeze(0).detach().cpu().numpy(),
                              "images/diff": torch.abs(gt_corr[0] - corr[0]).unsqueeze(0).detach().cpu().numpy(),
                              "images/max": max_corr[0].unsqueeze(0).detach().cpu().numpy(),
                              "images/max_diff": torch.abs(gt_corr[0] - max_corr[0]).unsqueeze(0).detach().cpu().numpy(),
                              "3D_corr_images/source": pc1_pv, "3D_corr_images/target": pc2_pv}
            log_images(test_writer, image_log_dict, epoch)
            test_writer.flush()

            print("Saving model ...")
            torch.save({"model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       os.path.join(log_dir, str(epoch).zfill(6) + '.pt'))

writer.close()
test_writer.close()

