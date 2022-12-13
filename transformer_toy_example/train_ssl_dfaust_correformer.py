
import numpy as np
import sys
import os
sys.path.append('../')
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import argparse
from data_ssl_noise import NoiseGenerator
from DfaustDataset import DfaustActionClipsDataset as Dataset
import visualization as vis
from models.correformer import CorreFormer
import models.correformer


def log_images(writer,
               log_dict, iter):
    for key in log_dict.keys():
        writer.add_image(key, log_dict[key], iter)
    writer.flush()


def log_scalars(writer, log_dict, iter):
    for key in log_dict.keys():
        writer.add_scalar(key, log_dict[key], iter)
    writer.flush()

def get_frame_pairs_train(points):
    # generate correspondance pairs by shuffling the points
    points2 = points.clone().detach()
    point_ids = torch.randperm(args.n_points).cuda()
    points2 = points2[:, point_ids, :]
    gt_corr = (point_ids.unsqueeze(-1) == torch.arange(args.n_points).cuda().unsqueeze(-2)).float().unsqueeze(0).repeat(
        [args.batch_size, 1, 1]).cuda()
    return points, points2, point_ids, gt_corr

def get_frame_pairs_test(points):
    # generate correspondance pairs by shifting the sequence by one frame
    points2 = torch.roll(points, -1, dims=1).clone().detach()
    points = points[:, 0:-1, :, :].reshape(-1, args.n_points, 3)  # remove first frame pair
    points2 = points2[:, 0:-1, :, :].reshape(-1, args.n_points, 3)  # remove first frame pair
    point_ids = torch.randperm(args.n_points).cuda()
    points2 = points2[:, point_ids, :]
    gt_corr = (point_ids.unsqueeze(-1) == torch.arange(args.n_points).cuda().unsqueeze(-2)).float().unsqueeze(0).repeat(
        [args.batch_size, 1, 1]).cuda()
    return points, points2, point_ids, gt_corr


parser = argparse.ArgumentParser()
parser.add_argument("--n_points", type=int, default=1024)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--dim", type=int, default=1024)
parser.add_argument("--n_heads", type=int, default=16)
parser.add_argument("--train_epochs", type=int, default=500000)
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/dfaust/', help='path to dataset')
parser.add_argument('--frames_per_clip', type=int, default=1, help='number of frames in a clip sequence')
parser.add_argument("--eval_steps", type=int, default=20)
parser.add_argument('--gender', type=str,
                    default='female', help='female | male | all indicating which subset of the dataset to use')
point_size = 25
args = parser.parse_args()

args.exp_name = f"ssl_N{args.n_points}_d{args.dim}h{args.n_heads}_lr{args.learning_rate}bs{args.batch_size}"
log_dir = "./log_ssl/" + args.exp_name
model_path = log_dir + "/model"
writer = SummaryWriter(os.path.join(log_dir, 'train'))
test_writer = SummaryWriter(os.path.join(log_dir, 'test'))

# Set up data
train_dataset = NoiseGenerator(args.n_points, radius=0.5, n_samples=512, sigma=0.3)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                                               pin_memory=True, shuffle=True, drop_last=True)
test_dataset = Dataset(args.dataset_path, frames_per_clip=args.frames_per_clip + 1, set='test', n_points=args.n_points,
                       shuffle_points='each', gender=args.gender)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                              pin_memory=True, drop_last=True)
test_enum = enumerate(test_dataloader, 0)

# set up model
model = CorreFormer(d_model=args.dim, nhead=args.n_heads, num_encoder_layers=6, num_decoder_layers=1,
                    dim_feedforward=1024)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
model = nn.DataParallel(model).cuda()

eval_steps = 0

for epoch in range(args.train_epochs):
    model.train()
    losses = []
    for batch_idx, data in enumerate(train_dataloader):

        points = data['points'].cuda()
        points, points2, point_ids, gt_corr = get_frame_pairs_train(points)
        out_dict = model(points, points2)
        out1, out2, corr, max_ind = out_dict['out1'], out_dict['out2'], out_dict['corr_mat'], out_dict['corr_idx21']

        loss = models.correformer.compute_corr_loss(gt_corr, corr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        true_corr = max_ind == point_ids
        correct = (true_corr).sum().detach().cpu().numpy()
        avg_acc = correct / (args.n_points*args.batch_size)
        iter = epoch * len(train_dataloader) + batch_idx

        loss_log_dict = {"acc": avg_acc,
                         "losses/total_loss": loss.detach().cpu().numpy()}
        log_scalars(writer, loss_log_dict, iter)

        print(f"Epoch {epoch} batch {batch_idx}: train loss {loss:.3f}")

        eval_steps = eval_steps + 1

        ################################################### run test ###########################################
        if eval_steps == args.eval_steps:
            eval_steps = 0

            with torch.no_grad():
                model.eval()  # Set model to evaluate mode
                test_batchind, test_data = next(test_enum)
                if test_batchind == len(test_dataloader) - 1:
                    test_enum = enumerate(test_dataloader, 0)
                test_points = test_data['points']
                test_points, test_points2, test_point_ids, test_gt_corr = get_frame_pairs_test(test_points)
                test_out_dict = model(test_points, test_points2)
                test_out1, test_out2, test_corr, test_max_ind = test_out_dict['out1'], test_out_dict['out2'], \
                    test_out_dict['corr_mat'], test_out_dict['corr_idx21']

                test_loss = models.correformer.compute_corr_loss(test_gt_corr, test_corr)
                print(f"Epoch {epoch} batch {batch_idx}: test loss {test_loss:.3f}")

                test_true_corr = test_max_ind == test_point_ids
                test_correct = (test_true_corr).sum().detach().cpu().numpy()
                test_avg_acc = test_correct / (args.n_points * args.batch_size)
                test_loss_log_dict = {"acc": test_avg_acc,
                                      "losses/total_loss": test_loss.detach().cpu().numpy()}
                log_scalars(test_writer, test_loss_log_dict, iter)
                model.train()

    if epoch % 5 == 0:

        max_corr = (max_ind[0].unsqueeze(-1) == torch.arange(args.n_points).cuda().unsqueeze(-2)).float().unsqueeze(0).repeat([args.batch_size, 1, 1]).cuda()
        pc1_pv = vis.get_pc_pv_image(points[0].detach().cpu().numpy(), text=None,
                                     color=np.arange(args.n_points), point_size=point_size).transpose(2, 0, 1)
        pc2_pv = vis.get_pc_pv_image(points2[0].detach().cpu().numpy(),  text=None,
                                     color=max_ind[0].detach().cpu().numpy(), point_size=point_size).transpose(2, 0, 1)
        pc_diff_pv = vis.get_pc_pv_image(points2[0].detach().cpu().numpy(),  text=None,
                                     color=torch.logical_not(true_corr[0]).detach().cpu().numpy(),
                                         point_size=point_size, cmap='jet').transpose(2, 0, 1)
        image_log_dict = {"images/GT": gt_corr[0].unsqueeze(0).detach().cpu().numpy(),
                          "images/corr": corr[0].unsqueeze(0).detach().cpu().numpy(),
                          "images/diff": torch.abs(gt_corr[0] - corr[0]).unsqueeze(0).detach().cpu().numpy(),
                          "images/max": max_corr[0].unsqueeze(0).detach().cpu().numpy(),
                          "images/max_diff": torch.abs(gt_corr[0] - max_corr[0]).unsqueeze(0).detach().cpu().numpy(),
                          "3D_corr_images/source": pc1_pv, "3D_corr_images/target": pc2_pv,
                          "3D_corr_images/diff": pc_diff_pv}
        log_images(writer, image_log_dict, epoch)
        print(f"Epoch {epoch}: train loss {loss:.3f}")


        # log test images
        test_max_corr = (test_max_ind[0].unsqueeze(-1) == torch.arange(args.n_points).cuda().unsqueeze(-2)).float().unsqueeze(
            0).repeat([args.batch_size, 1, 1]).cuda()
        test_pc1_pv = vis.get_pc_pv_image(test_points[0].detach().cpu().numpy(), text=None, color=np.arange(args.n_points),
                                     point_size=point_size).transpose(2, 0, 1)
        test_pc2_pv = vis.get_pc_pv_image(test_points2[0].detach().cpu().numpy(), text=None,
                                     color=test_max_ind[0].detach().cpu().numpy(), point_size=point_size).transpose(2, 0, 1)
        test_pc_diff_pv = vis.get_pc_pv_image(test_points2[0].detach().cpu().numpy(),  text=None,
                                     color=torch.logical_not(test_true_corr[0]).detach().cpu().numpy(),
                                         point_size=point_size, cmap='jet').transpose(2, 0, 1)
        test_image_log_dict = {"images/GT": test_gt_corr[0].unsqueeze(0).detach().cpu().numpy(),
                          "images/corr": test_corr[0].unsqueeze(0).detach().cpu().numpy(),
                          "images/diff": torch.abs(test_gt_corr[0] - test_corr[0]).unsqueeze(0).detach().cpu().numpy(),
                          "images/max": test_max_corr[0].unsqueeze(0).detach().cpu().numpy(),
                          "images/max_diff": torch.abs(test_gt_corr[0] - test_max_corr[0]).unsqueeze(0).detach().cpu().numpy(),
                          "3D_corr_images/source": test_pc1_pv, "3D_corr_images/target": test_pc2_pv,
                            "3D_corr_images/diff": test_pc_diff_pv}
        log_images(test_writer, test_image_log_dict, epoch)

        print("Saving model ...")
        torch.save({"model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()},
                   os.path.join(log_dir, str(epoch).zfill(6) + '.pt'))

writer.close()
test_writer.close()
