# Author: Yizhak Ben-Shabat (Itzik), 2022
# train 3DInAction

import os
import yaml
import argparse
import i3d_utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from models.pointnet import feature_transform_regularizer
from models import build_model
from datasets import build_dataloader

import wandb
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='./log/', help='path to model save dir')
parser.add_argument('--identifier', type=str, default='debug', help='unique run identifier')
parser.add_argument('--config', type=str, default='./configs/config_dfaust.yaml', help='path to yaml config file')
parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed')
args = parser.parse_args()


def main():
    cfg = yaml.safe_load(open(args.config))
    logdir = os.path.join(args.logdir, args.identifier)
    os.makedirs(logdir, exist_ok=True)

    wandb_run = wandb.init(project='DFAUST', entity='cgmlab', save_code=True)
    cfg['WANDB'] = {'id': wandb_run.id, 'project': wandb_run.project, 'entity': wandb_run.entity}

    with open(os.path.join(logdir, 'config.yaml'), 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

    wandb_run.name = args.identifier
    wandb.config.update(cfg)  # adds all the arguments as config variables
    wandb.run.log_code(".")
    # define our custom x axis metric
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("test/*", step_metric="train/step")

    # need to add argparse
    run(cfg, logdir)

def run(cfg, logdir):
    n_epochs = cfg['TRAINING']['n_epochs']
    lr = cfg['TRAINING']['lr']
    batch_size = cfg['TRAINING']['batch_size']
    refine, refine_epoch = cfg['TRAINING']['refine'], cfg['TRAINING']['refine_epoch']
    pretrained_model = cfg['TRAINING']['pretrained_model']
    pc_model = cfg['MODEL']['pc_model']
    frames_per_clip = cfg['DATA']['frames_per_clip']
    num_steps_per_update = cfg['TRAINING']['steps_per_update']
    save_every = cfg['save_every']

    if args.fix_random_seed:
        seed = cfg['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.system('cp %s %s' % (__file__, logdir))  # backup the current training file
    os.makedirs(os.path.join(logdir, 'models'), exist_ok=True)
    os.system('cp %s %s' % ('models/*.py', os.path.join(logdir, 'models')))  # backup the models files

    # build dataloader and dataset
    train_dataloader, train_dataset = build_dataloader(config=cfg, training=True, shuffle=False) # should be unshuffled because of sampler
    test_dataloader, test_dataset = build_dataloader(config=cfg, training=False, shuffle=True)

    num_classes = train_dataset.num_classes

    # build model
    model = build_model(cfg['MODEL'], num_classes, frames_per_clip)

    if pretrained_model is not None:
        checkpoints = torch.load(pretrained_model)
        model.load_state_dict(checkpoints["model_state_dict"])  # load trained model
        model.replace_logits(num_classes)

    if refine:
        if refine_epoch == 0:
            raise ValueError("You set the refine epoch to 0. No need to refine, just retrain.")
        refine_model_filename = os.path.join(logdir, str(refine_epoch).zfill(6)+'.pt')
        checkpoint = torch.load(refine_model_filename)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.cuda()
    model = nn.DataParallel(model)

    # define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1E-6)
    lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    if refine:
        lr_sched.load_state_dict(checkpoint["lr_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    steps = 0
    n_examples = 0
    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)
    refine_flag = True

    pbar = tqdm(total=n_epochs, desc='Training', dynamic_ncols=True)
    while steps <= n_epochs:
        if steps <= refine_epoch and refine and refine_flag:
            lr_sched.step()
            steps += 1
            n_examples += len(train_dataset.clip_set)
            continue
        else:
            refine_flag = False
        # Each epoch has a training and validation phase

        test_batchind = -1
        test_fraction_done = 0.0
        test_enum = enumerate(test_dataloader, 0)
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
        num_iter = 0
        optimizer.zero_grad()

        # Iterate over data.
        avg_acc = []
        loader_pbar = tqdm(total=len(train_dataloader), dynamic_ncols=True, leave=False)
        for train_batchind, data in enumerate(train_dataloader):
            num_iter += 1
            # get the inputs
            inputs, labels = data['points'], data['labels']
            inputs = inputs.permute(0, 1, 3, 2).cuda().requires_grad_().contiguous()
            labels = F.one_hot(labels.to(torch.int64), num_classes).permute(0, 2, 1).float().cuda()

            out_dict = model(inputs)
            per_frame_logits = out_dict['pred']

            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
            tot_loc_loss += loc_loss.item()

            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
            tot_cls_loss += cls_loss.item()
            loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
            if pc_model == 'pn1' or pc_model == 'pn1_4d': # TODO: what about 'pn1_4d_basic'?
                trans, trans_feat = out_dict['trans'], out_dict['trans_feat']
                loss += 0.001 * feature_transform_regularizer(trans) + 0.001 * feature_transform_regularizer(trans_feat)

            tot_loss += loss.item()
            loss.backward()

            acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), torch.argmax(labels, dim=1))
            avg_acc.append(acc.item())

            train_fraction_done = (train_batchind + 1) / train_num_batch

            if num_iter == num_steps_per_update or train_batchind == len(train_dataloader)-1:
                n_steps = num_steps_per_update
                if train_batchind == len(train_dataloader)-1:
                    n_steps = num_iter
                n_examples += batch_size*n_steps
                optimizer.step()
                optimizer.zero_grad()
                # log train losses
                log_dict = {
                    "train/step": n_examples,
                    "train/loss": tot_loss / n_steps,
                    "train/cls_loss": tot_cls_loss / n_steps,
                    "train/loc_loss": tot_loc_loss / n_steps,
                    "train/Accuracy": np.mean(avg_acc),
                    "train/lr":  optimizer.param_groups[0]['lr']
                }
                wandb.log(log_dict)

                num_iter = 0
                tot_loss = 0.

            if test_fraction_done <= train_fraction_done and test_batchind + 1 < test_num_batch:
                model.eval()
                test_batchind, data = next(test_enum)
                inputs, labels = data['points'], data['labels']
                inputs = inputs.permute(0, 1, 3, 2).cuda().requires_grad_().contiguous()
                labels = F.one_hot(labels.to(torch.int64), num_classes).permute(0, 2, 1).float().cuda()

                with torch.no_grad():
                    out_dict = model(inputs)
                    per_frame_logits = out_dict['pred']
                    # compute localization loss
                    loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                    # compute classification loss (with max-pooling along time B x C x T)
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                                  torch.max(labels, dim=2)[0])
                    loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                    if pc_model == 'pn1' or pc_model == 'pn1_4d': # TODO: what about 'pn1_4d_basic'?
                        trans, trans_feat = out_dict['trans'], out_dict['trans_feat']
                        loss += (0.001 * feature_transform_regularizer(trans) +
                                 0.001 * feature_transform_regularizer(trans_feat)) / num_steps_per_update
                    acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), torch.argmax(labels, dim=1))

                log_dict = {
                    "test/step": n_examples,
                    "test/loss": loss.item(),
                    "test/cls_loss": loc_loss.item(),
                    "test/loc_loss": cls_loss.item(),
                    "test/Accuracy": acc.item()
                }
                wandb.log(log_dict)
                test_fraction_done = (test_batchind + 1) / test_num_batch
                model.train()

            loader_pbar.update()
        loader_pbar.close()

        if steps % save_every == 0 or steps == n_epochs:
            # save model
            torch.save({"model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_state_dict": lr_sched.state_dict()},
                       os.path.join(logdir, str(steps).zfill(6) + '.pt'))

        steps += 1
        lr_sched.step()
        pbar.update()


if __name__ == '__main__':
    main()
