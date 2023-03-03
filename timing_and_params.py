import torch
from models import build_model
from datasets import build_dataloader

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    model_cfg = {'PATCHLET':
                     {'centroid_jitter': 0.001, 'sample_mode': 'nn', 'k': [16, 16, 16], 'npoints': [512, 128, None],
                      'temp_conv': 7, 'downsample_method': 'fps', 'radius': [None, None, None], 'type': 'skip_con',
                      'attn_num_heads': 4, 'extractor_type': 'strided', 'temporal_stride': 8,
                      'local_temp_convs': [8, 8, 8]
                      }
                 }

    model = PointNet2Patchlets(model_cfg, 10, n_frames=4)
    model_n_params = count_params(model)
    print(f'Patchlets number of parameters: {model_n_params:,}')

    model_2 = MSRAction(radius=0.3, nsamples=9, num_classes=10)
    model_2_n_params = count_params(model_2)
    print(f'MSRAction number of parameters: {model_2_n_params:,}')

if __name__ == '__main__':
    main()







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
parser.add_argument('--config', type=str, default='./configs/dfaust/config_dfaust.yaml', help='path to yaml config file')
parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed')
args = parser.parse_args()


def main():
    cfg = yaml.safe_load(open(args.config))
    logdir = os.path.join(args.logdir, args.identifier)
    os.makedirs(logdir, exist_ok=True)

    with open(os.path.join(logdir, 'config.yaml'), 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)



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


    # build dataloader and dataset
    test_dataloader, test_dataset = build_dataloader(config=cfg, training=False, shuffle=True)
    num_classes = test_dataset.num_classes

    # build model
    model = build_model(cfg['MODEL'], num_classes, frames_per_clip)

    model.cuda()
    model = nn.DataParallel(model)

    steps = 0
    n_examples = 0
    test_num_batch = len(test_dataloader)
    refine_flag = True

    pbar = tqdm(total=n_epochs, desc='Training', dynamic_ncols=True)
    while steps <= n_epochs:
        test_batchind = -1
        test_fraction_done = 0.0
        test_enum = enumerate(test_dataloader, 0)
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
        num_iter = 0

        # Iterate over data.
        avg_acc = []
        loader_pbar = tqdm(total=len(train_dataloader), dynamic_ncols=True, leave=False)
        for train_batchind, data in enumerate(train_dataloader):
            num_iter += 1
            # get the inputs
            inputs, labels, vid_idx, frame_pad = data['inputs'], data['labels'], data['vid_idx'], data['frame_pad']
            in_channel = cfg['MODEL'].get('in_channel', 3)
            inputs = inputs[:, :, 0:in_channel, :]
            inputs = inputs.cuda().requires_grad_().contiguous()
            labels = labels.cuda()

            out_dict = model(inputs)
            per_frame_logits = out_dict['pred']



            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
            tot_cls_loss += cls_loss.item()
            loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
            if pc_model == 'pn1' or pc_model == 'pn1_4d_basic':
                trans, trans_feat = out_dict['trans'], out_dict['trans_feat']
                loss += 0.001 * feature_transform_regularizer(trans) + 0.001 * feature_transform_regularizer(trans_feat)

            tot_loss += loss.item()


            acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), torch.argmax(labels, dim=1))
            avg_acc.append(acc.item())





            model.eval()
            test_batchind, data = next(test_enum)
            inputs, labels, vid_idx, frame_pad = data['inputs'], data['labels'], data['vid_idx'], data['frame_pad']
            in_channel = cfg['MODEL'].get('in_channel', 3)
            inputs = inputs[:, :, 0:in_channel, :]
            inputs = inputs.cuda().requires_grad_().contiguous()
            labels = labels.cuda()

            with torch.no_grad():
                out_dict = model(inputs)

            loader_pbar.update()
        loader_pbar.close()

        steps += 1
        pbar.update()


if __name__ == '__main__':
    main()






