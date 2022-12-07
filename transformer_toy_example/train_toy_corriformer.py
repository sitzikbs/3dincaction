# from modules import ISAB, PMA, SAB
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import argparse
from data_spheres import SphereGenerator
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('../')
import visualization as vis
from models.corriformer import CorriFormer
from utils import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--num_pts", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dim", type=int, default=1024)
parser.add_argument("--n_heads", type=int, default=16)
parser.add_argument("--train_epochs", type=int, default=500000)
args = parser.parse_args()
args.exp_name = f"set_N{args.num_pts}_d{args.dim}h{args.n_heads}_lr{args.learning_rate}bs{args.batch_size}"
log_dir = "./log/" + args.exp_name
model_path = log_dir + "/model"
writer = SummaryWriter(log_dir)


# Set up data
sphere_dataset = SphereGenerator(args.num_pts, 0.7, 512)
dataloader = torch.utils.data.DataLoader(sphere_dataset, batch_size=args.batch_size, num_workers=0,
                                               pin_memory=True, shuffle=True, drop_last=True)

# set up model
model = CorriFormer(d_model=args.dim, nhead=args.n_heads, num_encoder_layers=6, num_decoder_layers=1, dim_feedforward=1024)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
model = nn.DataParallel(model)
model = model.cuda()
# criterion = torch.nn.L1Loss(reduction='none')
criterion = torch.nn.MSELoss(reduction='none')

for epoch in range(args.train_epochs):
    model.train()
    losses, total, correct = [], 0, 0
    for batch_idx, points in enumerate(dataloader):
        points = points.cuda()
        points2 = ((torch.randn([args.batch_size, 1, 1]).cuda() - 0.5) * 2 * 0.3 + 1.0) * points.detach().clone()

        point_ids = torch.randperm(args.num_pts).cuda()
        points2 = points2[:, point_ids, :]

        gt_corr = (point_ids.unsqueeze(-1) == torch.arange(args.num_pts).cuda().unsqueeze(-2)).float().unsqueeze(0).repeat([args.batch_size, 1, 1]).cuda()

        out1 = model(points)
        out2 = model(points2)
        # out2 = out1[:, point_ids, :] #debugging

        corr = F.softmax(cosine_similarity(out2, out1), dim=-1)

        l1_loss = criterion(gt_corr, corr)
        l1_mask = torch.max(gt_corr, 1.0*(torch.rand(args.batch_size, args.num_pts, args.num_pts).cuda() < gt_corr.mean()))
        l1_loss = (l1_mask * l1_loss).mean()

        self_corr = cosine_similarity(out1, out1)
        self_mask = torch.max(torch.eye(args.num_pts, args.num_pts).cuda(),
                            1.0 * (torch.rand(args.num_pts, args.num_pts).cuda() < torch.eye(args.num_pts).cuda().mean()))
        loss_iden = criterion(torch.eye(args.num_pts).unsqueeze(0).expand(args.batch_size, -1, -1).cuda(), self_corr)
        loss_iden = (self_mask * loss_iden).mean()
        # loss = l1_loss + loss_iden
        loss = l1_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        max_val, max_ind = torch.max(corr, dim=-1)
        total += args.num_pts*args.batch_size
        correct += (max_ind == point_ids).sum().item()
        avg_acc = correct / total
        iter = epoch * len(sphere_dataset) + batch_idx

        writer.add_scalar("acc", avg_acc, iter)



        writer.add_scalar("losses/l1_loss", l1_loss.detach().cpu().numpy(), iter)
        writer.add_scalar("losses/identity_loss", loss_iden.detach().cpu().numpy(), iter)
        writer.add_scalar("losses/total_loss", loss.detach().cpu().numpy(), iter)

        # wandb_run.log({"train_loss": loss.detach().cpu().numpy()})
        print(f"Epoch {epoch} batch {batch_idx}: train loss {loss:.3f}")

    if epoch % 100 == 0:
        writer.add_image("images/GT", gt_corr[0].unsqueeze(0).detach().cpu().numpy(), epoch)
        writer.add_image("images/corr", corr[0].unsqueeze(0).detach().cpu().numpy(), epoch)
        writer.add_image("images/diff", torch.abs(gt_corr[0] - corr[0]).unsqueeze(0).detach().cpu().numpy(), epoch)
        max_corr = (max_ind[0].unsqueeze(-1) == torch.arange(args.num_pts).cuda().unsqueeze(-2)).float().unsqueeze(0).repeat([args.batch_size, 1, 1]).cuda()
        writer.add_image("images/max", max_corr[0].unsqueeze(0).detach().cpu().numpy(), epoch)
        writer.add_image("images/max_diff", torch.abs(gt_corr[0] - max_corr[0]).unsqueeze(0).detach().cpu().numpy(), epoch)

        pc1_pv = vis.get_pc_pv_image(points[0].detach().cpu().numpy(), text=None, color=np.arange(args.num_pts))
        pc2_pv = vis.get_pc_pv_image(points2[0].detach().cpu().numpy(), text=None, color=max_ind[0].detach().cpu().numpy())
        writer.add_image("3D_corr_images/source", pc1_pv.transpose(2, 0, 1), epoch)
        writer.add_image("3D_corr_images/target", pc2_pv.transpose(2, 0, 1), epoch)
        print(f"Epoch {epoch}: train loss {loss:.3f}")
