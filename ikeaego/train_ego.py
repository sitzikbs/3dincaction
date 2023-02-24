# Author: Yizhak Ben-Shabat (Itzik), 2023
# train action recognition on the ikea ASM dataset

import os
import sys
import yaml
sys.path.append('../')
import argparse
import i3d_utils as utils
import utils as point_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np
from IKEAEgoDatasetClips import IKEAEgoDatasetClips as Dataset
from tensorboardX import SummaryWriter

from models.pytorch_i3d import InceptionI3d
from models.pointnet import PointNet4D, feature_transform_regularizer, PointNet1, PointNet1Basic
from models.pointnet2_cls_ssg import PointNetPP4D
from models.pytorch_3dmfv import FourDmFVNet
from models.pointnet2_cls_ssg import PointNet2, PointNetPP4D, PointNet2Basic
from models.patchlets import PointNet2Patchlets

import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='./log/', help='path to model save dir')
parser.add_argument('--identifier', type=str, default='debug', help='unique run identifier')
parser.add_argument('--config', type=str, default='./configs/config_ikeaego.yaml', help='path to configuration yaml file')
args = parser.parse_args()


def run(cfg, logdir):
    max_steps = cfg['TRAINING']['n_epochs'] + 1
    lr = cfg['TRAINING']['lr']
    dataset_path = cfg['DATA']['dataset_path']
    batch_size = cfg['TRAINING']['batch_size']
    refine, refine_epoch = cfg['TRAINING']['refine'], cfg['TRAINING']['refine_epoch']
    pretrained_model = cfg['TRAINING']['pretrained_model']
    pc_model = cfg['MODEL']['pc_model']
    num_steps_per_update = cfg['TRAINING']['steps_per_update']
    in_channel = cfg['DATA']['in_channel']
    assert in_channel == 3 or in_channel == 6 or in_channel == 9

    os.system('cp %s %s' % (__file__, logdir))  # backup the current training file
    os.system('cp %s %s' % ('../models/pytorch_i3d.py', logdir))  # backup the models files
    os.system('cp %s %s' % ('../models/pointnet.py', logdir))  # backup the models files
    os.system('cp %s %s' % ('../models/pointnet2_cls_ssg.py', logdir))  # backup the models files
    os.system('cp %s %s' % ('../models/pytorch_3dmfv.py', logdir))  # backup the models files
    os.system('cp %s %s' % ('../models/patchlets.py', logdir))  # backup the models files

    # TODO add transforms to training

    train_dataset = Dataset(dataset_path, set='train')
    print("Number of clips in the dataset:{}".format(len(train_dataset)))
    weights = utils.make_weights_for_balanced_classes(train_dataset.clip_set, train_dataset.clip_label_count)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                                   num_workers=8, pin_memory=True)


    test_dataset = Dataset(dataset_path, set='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                  pin_memory=True)
    num_classes = train_dataset.num_classes
    frames_per_clip = train_dataset.frames_per_clip

    if pc_model == 'pn1':
        model = PointNet1(k=num_classes, feature_transform=True, in_d=in_channel)
    elif pc_model == 'pn1_4d':
        model = PointNet4D(k=num_classes, feature_transform=True, n_frames=frames_per_clip)
    elif pc_model == 'pn1_4d_basic':
        model = PointNet1Basic(k=num_classes, feature_transform=True, n_frames=frames_per_clip, in_d=in_channel)
    elif pc_model == 'pn2':
        model = PointNet2(num_class=num_classes, n_frames=frames_per_clip, in_channel=in_channel)
    elif pc_model == 'pn2_4d':
        model = PointNetPP4D(num_class=num_classes, n_frames=frames_per_clip)
    elif pc_model == 'pn2_4d_basic':
        model = PointNet2Basic(num_class=num_classes, n_frames=frames_per_clip, in_channel=in_channel)
    elif pc_model == 'pn2_patchlets':
        model = PointNet2Patchlets(cfg=cfg['MODEL']['PATCHLET'], num_class=num_classes, n_frames=frames_per_clip,
                                   in_channel=in_channel)
    elif pc_model == '3dmfv':
        model = FourDmFVNet(n_gaussians=cfg['MODEL']['3DMFV']['n_gaussians'], num_classes=num_classes, n_frames=frames_per_clip)
    else:
        raise ValueError("point cloud architecture not supported. Check the pc_model input")


    if pretrained_model is not None:
        checkpoints = torch.load(pretrained_model)
        model.load_state_dict(checkpoints["model_state_dict"])  # load trained model
        model.replace_logits(num_classes)

    else:
        for name, param in model.named_parameters():
            param.requires_grad = True

    if refine:
        if refine_epoch == 0:
            raise ValueError("You set the refine epoch to 0. No need to refine, just retrain.")
        refine_model_filename = os.path.join(logdir, str(refine_epoch).zfill(6)+'.pt')
        checkpoint = torch.load(refine_model_filename)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.cuda()
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1E-6)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40])

    if refine:
        lr_sched.load_state_dict(checkpoint["lr_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    train_writer = SummaryWriter(os.path.join(logdir, 'train'))
    test_writer = SummaryWriter(os.path.join(logdir, 'test'))

    steps = 0
    n_examples = 0
    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)
    refine_flag = True

    while steps < max_steps:#for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)
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
        for train_batchind, data in enumerate(train_dataloader):

            num_iter += 1
            # get the inputs
            inputs, labels, vid_idx, frame_pad = data
            inputs = inputs.cuda().requires_grad_()
            labels = labels.cuda()

            inputs = inputs[:, :, 0:in_channel, :]
            out_dict = model(inputs)
            per_frame_logits = out_dict['pred']

            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
            tot_loc_loss += loc_loss.item()

            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
            tot_cls_loss += cls_loss.item()
            loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
            if pc_model == 'pn1':
                trans, trans_feat = out_dict['trans'], out_dict['trans_feat']
                loss = loss + 0.001*feature_transform_regularizer(trans) + 0.001*feature_transform_regularizer(trans_feat)

            tot_loss += loss.item()
            loss.backward()

            acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), torch.argmax(labels, dim=1))

            avg_acc.append(acc.item())

            train_fraction_done = (train_batchind + 1) / train_num_batch
            print('[{}] train Acc: {}, Loss: {:.4f} [{} / {}]'.format(steps, acc.item(), loss.item(), train_batchind, len(train_dataloader)))
            if (num_iter == num_steps_per_update or train_batchind == len(train_dataloader)-1) :
                n_steps = num_steps_per_update
                if train_batchind == len(train_dataloader)-1:
                    n_steps = num_iter
                n_examples += batch_size*n_steps
                print('updating the model...')
                print('train Total Loss: {:.4f}'.format(tot_loss / n_steps))
                optimizer.step()
                optimizer.zero_grad()

                # log train losses
                log_dict = {
                    "train/step": n_examples,
                    "train/loss": tot_loss / n_steps,
                    "train/cls_loss": tot_cls_loss / n_steps,
                    "train/loc_loss": tot_loc_loss / n_steps,
                    "train/Accuracy": np.mean(avg_acc),
                    "train/lr":  optimizer.param_groups[0]['lr'] }
                wandb.log(log_dict)
                train_writer.add_scalar('loss', tot_loss / n_steps, n_examples)
                train_writer.add_scalar('cls loss', tot_cls_loss / n_steps, n_examples)
                train_writer.add_scalar('loc loss', tot_loc_loss / n_steps, n_examples)
                train_writer.add_scalar('Accuracy', np.mean(avg_acc), n_examples)
                train_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], n_examples)

                num_iter = 0
                tot_loss = 0.

            if test_fraction_done <= train_fraction_done and test_batchind + 1 < test_num_batch:
                model.train(False)  # Set model to evaluate mode
                test_batchind, data = next(test_enum)
                inputs, labels, vid_idx, frame_pad = data
                inputs = inputs.cuda().requires_grad_().contiguous()
                inputs = inputs[:, :, 0:in_channel, :]
                labels = labels.cuda()

                with torch.no_grad():
                    out_dict = model(inputs)
                    per_frame_logits = out_dict['pred']
                    if pc_model == 'pn1':
                        trans, trans_feat = out_dict['trans'], out_dict['trans_feat']

                    # compute localization loss
                    loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)

                    # compute classification loss (with max-pooling along time B x C x T)
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                                  torch.max(labels, dim=2)[0])

                    loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                    if pc_model == 'pn1':
                        loss = loss + (0.001 * feature_transform_regularizer(trans) + 0.001 * feature_transform_regularizer(
                            trans_feat)) / num_steps_per_update
                    acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), torch.argmax(labels, dim=1))


                print('[{}] test Acc: {}, Loss: {:.4f} [{} / {}]'.format(steps, acc.item(), loss.item(), test_batchind,
                                                                     len(test_dataloader)))
                log_dict = {
                    "test/step": n_examples,
                    "test/loss": loss.item(),
                    "test/cls_loss": loc_loss.item(),
                    "test/loc_loss": cls_loss.item(),
                    "test/Accuracy": acc.item()}
                wandb.log(log_dict)
                test_writer.add_scalar('loss', loss.item(), n_examples)
                test_writer.add_scalar('cls loss', loc_loss.item(), n_examples)
                test_writer.add_scalar('loc loss', cls_loss.item(), n_examples)
                test_writer.add_scalar('Accuracy', acc.item(), n_examples)
                test_fraction_done = (test_batchind + 1) / test_num_batch
                model.train(True)

        if steps % 5 == 0:
            # save model
            print("Saving model ...")
            torch.save({"model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_state_dict": lr_sched.state_dict()},
                       os.path.join(logdir, str(steps).zfill(6) + '.pt'))

        steps += 1
        lr_sched.step()
    train_writer.close()
    test_writer.close()


if __name__ == '__main__':
    cfg = yaml.safe_load(open(args.config))
    logdir = os.path.join(args.logdir, args.identifier)
    wandb_run = wandb.init(project='IKEA EGO', save_code=True)

    os.makedirs(logdir, exist_ok=True)
    cfg['WANDB'] = {'id': wandb_run.id, 'project': wandb_run.project, 'entity': wandb_run.entity}
    with open(os.path.join(logdir, 'config.yaml'), 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

    wandb_run.name = args.identifier
    wandb.config.update(cfg)  # adds all of the arguments as config variables
    wandb.run.log_code("../")
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("test/*", step_metric="train/step")

    # need to add argparse
    print("Starting training ...")
    run(cfg, logdir)
