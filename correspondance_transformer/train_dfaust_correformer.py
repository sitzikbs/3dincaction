
import numpy as np
import sys
import os
sys.path.append('../')
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import argparse
# from data_spheres import SphereGenerator
from DfaustDataset import DfaustActionClipsDataset as Dataset
import visualization as vis
from models.correformer import CorreFormer

import models.correformer
import pc_transforms as transforms
from utils import ScalarScheduler

def log_images(writer,
               log_dict, iter):
    for key in log_dict.keys():
        writer.add_image(key, log_dict[key], iter)
    writer.flush()


def log_scalars(writer, log_dict, iter):
    for key in log_dict.keys():
        writer.add_scalar(key, log_dict[key], iter)
    writer.flush()

def get_frame_pairs(points):
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
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--dim", type=int, default=1024)
parser.add_argument("--d_feedforward", type=int, default=1024)
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--train_epochs", type=int, default=500000)
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/dfaust/', help='path to dataset')
parser.add_argument('--aug', type=str, nargs='+',
                    default=['none'], help='list of augmentations to apply: scale, rotate, translate, jitter')
parser.add_argument('--frames_per_clip', type=int, default=1, help='number of frames in a clip sequence')
parser.add_argument("--eval_steps", type=int, default=10)
parser.add_argument('--gender', type=str,
                    default='all', help='female | male | all indicating which subset of the dataset to use')
parser.add_argument('--nn_sample_ratio', type=int,
                    default=1.0, help='sample nearest neighbor correct corresondences, if 1 takes all points')
parser.add_argument('--transformer_type', type=str,
                    default='none', help='plr | ptr | none - use point transformer layer (plr)'
                                        ' or point transformer full segmentation architecture (ptr)'
                                        'or none which is the default pytorch transformer implementation')
parser.add_argument('--loss_type', type=str,
                    default='ce', help='ce | l2 | ce_bbl indicating the loss type ')
parser.add_argument('--cat_points', dest='cat_points', action='store_false')
parser.set_defaults(cat_points=True)
parser.add_argument('--exp_id', type=str,
                    default='debug_noreg_norm_ce_new', help='a unique identifier to append to the experiment name')

point_size = 25
sigma = ScalarScheduler(init_value=0.005, steps=5, increment=0.0)
args = parser.parse_args()
args.exp_name = f"dfaust_N{args.n_points}ff{args.d_feedforward}_d{args.dim}h{args.n_heads}_ttype{args.transformer_type}lr{args.learning_rate}bs{args.batch_size}{args.exp_id}"
log_dir = "./log/" + args.exp_name
model_path = log_dir + "/model"
writer = SummaryWriter(os.path.join(log_dir, 'train'))
test_writer = SummaryWriter(os.path.join(log_dir, 'test'))
os.system('cp %s %s' % (__file__, log_dir))  # backup the current training file
os.system('cp %s %s' % ('../models/correformer.py', log_dir))  # backup the models files
params_filename = os.path.join(log_dir, 'params.pth')  # backup parameters file
torch.save(args, params_filename)

# Set up data
train_dataset = Dataset(args.dataset_path, frames_per_clip=args.frames_per_clip + 1, set='train', n_points=args.n_points,
                        shuffle_points='fps_each', data_augmentation=args.aug, gender=args.gender,
                        nn_sample_ratio=args.nn_sample_ratio)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8,
                                               pin_memory=True, shuffle=True, drop_last=True)
test_dataset = Dataset(args.dataset_path, frames_per_clip=args.frames_per_clip + 1, set='test', n_points=args.n_points,
                       shuffle_points='fps_each', gender=args.gender)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                              pin_memory=True, drop_last=True)
test_enum = enumerate(test_dataloader, 0)

# set up model
model = CorreFormer(d_model=args.dim, nhead=args.n_heads, num_encoder_layers=6, num_decoder_layers=1,
                    dim_feedforward=args.d_feedforward, transformer_type=args.transformer_type, twosided=False,
                    n_points=args.n_points, loss_type=args.loss_type, cat_points=args.cat_points)
optimizer = torch.optim.Adam([{'params': model.pointencoder.parameters()},
                              {'params': model.transformer.parameters(), 'lr': args.learning_rate}], lr=args.learning_rate)
model = nn.DataParallel(model).cuda()

eval_steps = 0

for epoch in range(args.train_epochs):
    sigma.step()
    model.train()
    losses = []
    for batch_idx, data in enumerate(train_dataloader):

        points = data['points'].cuda()
        points, points2, point_ids, gt_corr = get_frame_pairs(points)
        if 'jitter' in args.aug and sigma.value() > 0:
            points = transforms.jitter_point_cloud_torch(points, sigma=sigma.value(), clip=5*sigma.value())
            points2 = transforms.jitter_point_cloud_torch(points2, sigma=sigma.value(), clip=5*sigma.value())

        out_dict = model(points, points2)
        out1, out2, corr, max_ind = out_dict['out1'], out_dict['out2'], out_dict['corr_mat'], out_dict['corr_idx21']
        point_features1, point_features2 = out_dict['point_features1'], out_dict['point_features2']
        sim_mat = out_dict['sim_mat']

        loss, loss_dict = model.module.compute_corr_loss(gt_corr, corr, point_ids, out1, out2, sim_mat,
                                              point_features1, point_features2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        true_corr = max_ind == point_ids
        correct = (true_corr).sum().detach().cpu().numpy()
        avg_acc = correct / (args.n_points*args.batch_size)
        iter = epoch * len(train_dataloader) + batch_idx

        loss_log_dict = loss_dict
        loss_log_dict['acc'] = avg_acc
        # loss_log_dict = {"acc": avg_acc,
        #                  "losses/total_loss": loss.detach().cpu().numpy()}
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
                test_points, test_points2, test_point_ids, test_gt_corr = get_frame_pairs(test_points)
                test_out_dict = model(test_points, test_points2)
                test_out1, test_out2, test_corr, test_max_ind = test_out_dict['out1'], test_out_dict['out2'], \
                    test_out_dict['corr_mat'], test_out_dict['corr_idx21']
                test_sim_mat = out_dict['sim_mat']
                test_point_features1, test_point_features2 = out_dict['point_features1'], out_dict['point_features2']
                test_loss, test_loss_dict = model.module.compute_corr_loss(test_gt_corr, test_corr,
                                                           test_point_ids, test_out1, test_out2, test_sim_mat,
                                                           test_point_features1, test_point_features2)
                print(f"Epoch {epoch} batch {batch_idx}: test loss {test_loss:.3f}")

                test_true_corr = test_max_ind == test_point_ids
                test_correct = (test_true_corr).sum().detach().cpu().numpy()
                test_avg_acc = test_correct / (args.n_points * args.batch_size)

                test_loss_log_dict = test_loss_dict
                test_loss_log_dict['acc'] = test_avg_acc
                # test_loss_log_dict = {"acc": test_avg_acc,
                #                       "losses/total_loss": test_loss.detach().cpu().numpy()}
                log_scalars(test_writer, test_loss_log_dict, iter)
                model.train()

    if epoch % 50 == 0: # log images every 50 epochs

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


        # save model every 100 epochs
        if epoch % 25 == 0:
            print("Saving model ...")
            torch.save({"model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       os.path.join(log_dir, str(epoch).zfill(6) + '.pt'))

writer.close()
test_writer.close()

