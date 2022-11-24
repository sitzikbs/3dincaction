# Author: Yizhak Ben-Shabat (Itzik), 2022
# train 3DInAction on Dfaust dataset

import os

import argparse
import i3d_utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from DfaustDataset import DfaustActionClipsDataset as Dataset
from tensorboardX import SummaryWriter


from models.pointnet import PointNet4D, feature_transform_regularizer, PointNet1
from models.pointnet2_cls_ssg import PointNet2, PointNetPP4D
from models.pytorch_3dmfv import FourDmFVNet


parser = argparse.ArgumentParser()
# parser.add_argument('--mode', type=str, default='rgb', help='rgb or flow')
parser.add_argument('--pc_model', type=str, default='pn1', help='which model to use for point cloud processing: pn1 | pn2 ')
parser.add_argument('--steps_per_update', type=int, default=20, help='number of steps per backprop update')
parser.add_argument('--frames_per_clip', type=int, default=16, help='number of frames in a clip sequence')
parser.add_argument('--batch_size', type=int, default=8, help='number of clips per batch')
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs to train')
parser.add_argument('--n_points', type=int, default=1024, help='number of points in a point cloud')
parser.add_argument('--logdir', type=str, default='./log/pn2_4d_f32_p1024_shuffle_once/', help='path to model save dir')
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/dfaust/', help='path to dataset')
parser.add_argument('--refine', action="store_true", help='flag to refine the model')
parser.add_argument('--refine_epoch', type=int, default=0, help='refine model from this epoch')
parser.add_argument('--pretrained_model', type=str, default=None, help='path to pretrained model')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--n_gaussians', type=int, default=8, help='number of gaussians for 3DmFV representation')
parser.add_argument('--shuffle_points', type=str, default='once', help='once | each | none shuffle the input points '
                                                                       'at initialization | for each batch example | no shufll')
parser.add_argument('--sampler', type=str, default='weighted', help='weighted | none how to sample the clips ')
args = parser.parse_args()


def run(init_lr=0.001, max_steps=64e3, frames_per_clip=16, dataset_path='/home/sitzikbs/Datasets/dfaust/',
        logdir='', batch_size=8, refine=False, refine_epoch=0,
        pretrained_model='charades', steps_per_update=1, pc_model='pn1'):

    os.makedirs(logdir, exist_ok=True)
    os.system('cp %s %s' % (__file__, logdir))  # backup the current training file
    os.system('cp %s %s' % ('./models/pointnet.py', logdir))  # backup the models files
    os.system('cp %s %s' % ('./models/pointnet2_cls_ssg.py', logdir))  # backup the models files
    os.system('cp %s %s' % ('./models/pytorch_3dmfv.py', logdir))  # backup the models files

    # setup dataset
    # train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
    #                                        videotransforms.RandomHorizontalFlip()])
    # train_transforms = transforms.Compose([transforms.RandomCrop(224),
    #                                        transforms.RandomHorizontalFlip(p=0.5)])
    # test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    # test_transforms = transforms.Compose([transforms.CenterCrop(224)])

    train_dataset = Dataset(dataset_path, frames_per_clip=frames_per_clip, set='train', n_points=args.n_points,
                            shuffle_points=args.shuffle_points)
    print("Number of clips in the trainingset:{}".format(len(train_dataset)))

    if args.sampler == 'weighted':
        weights = train_dataset.make_weights_for_balanced_classes()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                                                   pin_memory=True, drop_last=True, sampler=sampler)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                                                   pin_memory=True, shuffle=True, drop_last=True)

    test_dataset = Dataset(dataset_path, frames_per_clip=frames_per_clip, set='test', n_points=args.n_points,
                           shuffle_points=args.shuffle_points)
    print("Number of clips in the testset:{}".format(len(test_dataset)))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                  pin_memory=True)
    num_classes = train_dataset.action_dataset.num_classes

    if pc_model == 'pn1':
        model = PointNet1(k=num_classes, feature_transform=True)
    elif pc_model == 'pn1_4d':
        model = PointNet4D(k=num_classes, feature_transform=True, n_frames=frames_per_clip)
    elif pc_model == 'pn2':
        model = PointNet2(num_class=num_classes, n_frames=frames_per_clip)
    elif pc_model == 'pn2_4d':
        model = PointNetPP4D(num_class=num_classes, n_frames=frames_per_clip)
    elif pc_model == '3dmfv':
        model = FourDmFVNet(n_gaussians=args.n_gaussians, num_classes=num_classes, n_frames=frames_per_clip)
    else:
        raise ValueError("point cloud architecture not supported. Check the pc_model input")


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

    lr = init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1E-6)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40])

    if refine:
        lr_sched.load_state_dict(checkpoint["lr_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    train_writer = SummaryWriter(os.path.join(logdir, 'train'))
    test_writer = SummaryWriter(os.path.join(logdir, 'test'))

    num_steps_per_update = steps_per_update # 4 * 5 # accum gradient - try to have number of examples per update match original code 8*5*4
    steps = 0

    # train it
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
            inputs, labels = data['points'], data['labels']
            inputs = inputs.permute(0, 1, 3, 2).cuda().requires_grad_().contiguous()
            labels = F.one_hot(labels.to(torch.int64), num_classes).permute(0, 2, 1).float().cuda()

            # inputs = inputs[:, :, 0:3, :]
            # t = inputs.size(1)
            out_dict = model(inputs)
            per_frame_logits = out_dict['pred']
            if pc_model == 'pn1':
                trans, trans_feat = out_dict['trans'], out_dict['trans_feat']
            # per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear', align_corners=True)


            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
            tot_loc_loss += loc_loss.item()

            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
            tot_cls_loss += cls_loss.item()
            loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
            if pc_model == 'pn1' or pc_model == 'pn1_4d':
                loss = loss + 0.001*feature_transform_regularizer(trans) + 0.001*feature_transform_regularizer(trans_feat)

            tot_loss += loss.item()
            loss.backward()

            acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), torch.argmax(labels, dim=1))
            # acc = utils.accuracy(per_frame_logits, labels)

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
                inputs, labels = data['points'], data['labels']
                inputs = inputs.permute(0, 1, 3, 2).cuda().requires_grad_().contiguous()
                labels = F.one_hot(labels.to(torch.int64), num_classes).permute(0, 2, 1).float().cuda()

                with torch.no_grad():

                    # inputs = inputs[:, :, 0:3, :]
                    # t = inputs.size(1)
                    out_dict = model(inputs)
                    per_frame_logits = out_dict['pred']
                    if pc_model == 'pn1':
                        trans, trans_feat = out_dict['trans'], out_dict['trans_feat']
                    # per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear', align_corners=True)


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
                       logdir + str(steps).zfill(6) + '.pt')

        steps += 1
        lr_sched.step()
    train_writer.close()
    test_writer.close()


if __name__ == '__main__':
    # need to add argparse
    print("Starting training ...")
    run(init_lr=args.lr, dataset_path=args.dataset_path, logdir=args.logdir, max_steps=args.n_epochs+1,
        batch_size=args.batch_size,  refine=args.refine, refine_epoch=args.refine_epoch,
        pretrained_model=args.pretrained_model, steps_per_update=args.steps_per_update,
        frames_per_clip=args.frames_per_clip, pc_model=args.pc_model)
