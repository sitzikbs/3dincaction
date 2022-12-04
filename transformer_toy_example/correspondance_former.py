from modules import ISAB, PMA, SAB
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import argparse
from data_spheres import SphereGenerator
import numpy as np
# import wandb


parser = argparse.ArgumentParser()
parser.add_argument("--num_pts", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--n_anc", type=int, default=16)
parser.add_argument("--train_epochs", type=int, default=500000)
args = parser.parse_args()
args.exp_name = f"set_N{args.num_pts}_d{args.dim}h{args.n_heads}i{args.n_anc}_lr{args.learning_rate}bs{args.batch_size}"
log_dir = "./log/" + args.exp_name
model_path = log_dir + "/model"
writer = SummaryWriter(log_dir)

class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_output=40,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze()

def cosine_similarity(x, y):
  # Compute the dot product between the two batches
  dot = torch.bmm(x, y.transpose(2, 1))

  # Compute the norms of the two batches
  norm1 = torch.norm(x, dim=-1)
  norm2 = torch.norm(y, dim=-1)

  # Compute the cosine similarity
  sim = dot.abs() / (norm1.unsqueeze(-1) * norm2.unsqueeze(-2))

  # Return the cosine similarity
  return sim


# wandb_run = wandb.init()
# wandb.run.name = args.exp_name
# Set up data
sphere_dataset = SphereGenerator(args.num_pts, 0.5, 32)
dataloader = torch.utils.data.DataLoader(sphere_dataset, batch_size=args.batch_size, num_workers=0,
                                               pin_memory=True, shuffle=True, drop_last=True)

# set up model
# model = SetTransformer(dim_hidden=args.dim, num_heads=args.n_heads, num_inds=args.n_anc,
#                        num_outputs=args.num_pts, dim_output=64)
model = torch.nn.Transformer(d_model=3, nhead=3, num_encoder_layers=4, num_decoder_layers=6, dim_feedforward=256)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
model = nn.DataParallel(model)
model = model.cuda()
criterion = torch.nn.L1Loss(reduction='none')


for epoch in range(args.train_epochs):
    model.train()
    losses, total, correct = [], 0, 0
    for batch_idx, points in enumerate(dataloader):
        points = points.cuda()
        points2 = points.detach().clone()

        point_ids = torch.randperm(args.num_pts).cuda()
        points2 = points2[:, point_ids, :]

        gt_corr = (point_ids.unsqueeze(-1) == torch.arange(args.num_pts).cuda().unsqueeze(-2)).float().unsqueeze(0).repeat([args.batch_size, 1, 1]).cuda()

        # out1 = model(points)
        # out2 = model(points2)
        # out2 = out1[:, point_ids, :] #debugging

        out1 = model(points, points)
        out2 = model(points2, points2)

        # out1 = torch.nn.functional.softmax(out1, dim=-1)
        # out2 = torch.nn.functional.softmax(out2, dim=-1)
        corr = torch.nn.functional.softmax(cosine_similarity(out2, out1), dim=-1)

        l1_loss = criterion(gt_corr, corr)
        l1_mask = torch.max(gt_corr, 1.0*(torch.rand(args.batch_size, args.num_pts, args.num_pts).cuda() < gt_corr.mean()))
        l1_loss = (l1_mask * l1_loss).mean()

        self_corr = cosine_similarity(out1, out1)
        self_mask = torch.max(torch.eye(args.num_pts).cuda(),
                            1.0 * (torch.rand(args.num_pts, args.num_pts).cuda() < torch.eye(args.num_pts).cuda().mean()))
        loss_iden = criterion(torch.eye(args.num_pts).cuda(), self_corr)
        loss_iden = (self_mask * loss_iden).mean()
        loss = l1_loss + loss_iden
        # loss = l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        max_val, max_ind = torch.max(corr, dim=-1)
        total += args.num_pts*args.batch_size
        correct += (max_ind == point_ids).sum().item()
        avg_acc = correct / total
        iter = epoch * len(sphere_dataset) + batch_idx

        writer.add_scalar("acc", avg_acc, iter)
        # wandb_run.log({"acc": avg_acc})


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
        print(f"Epoch {epoch}: train loss {loss:.3f}")

        # wandb_run.log({"images/GT": wandb.Image(PIL.Image.fromarray(gt_corr[0].unsqueeze(-1).detach().cpu().numpy()))})