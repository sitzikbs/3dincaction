# Author: Yizhak Ben-Shabat (Itzik), 2022
# test 3DInAction on Dfaust dataset

import os
import argparse
import i3d_utils
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import build_dataloader
import random
import yaml
from models import build_model_from_logdir

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='./log/', help='path to model save dir')
parser.add_argument('--identifier', type=str, default='debug', help='unique run identifier')
parser.add_argument('--model_ckpt', type=str, default='000000.pt', help='checkpoint to load')
parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed')
args = parser.parse_args()


def run(cfg, logdir, model_path, output_path):
    batch_size = cfg['TESTING']['batch_size']
    frames_per_clip = cfg['DATA']['frames_per_clip']
    subset = cfg['TESTING']['set']
    pred_output_filename = os.path.join(output_path, subset + '_pred.npy')
    json_output_filename = os.path.join(output_path, subset + '_action_segments.json')
    data_name = cfg['DATA']['name']

    if args.fix_random_seed:
        seed = cfg['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    test_dataloader, test_dataset = build_dataloader(config=cfg, training=False, shuffle=False)
    num_classes = test_dataset.num_classes

    # setup the model
    model = build_model_from_logdir(logdir, cfg['MODEL'], num_classes, frames_per_clip)

    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints["model_state_dict"])  # load trained model
    model.cuda()
    model = nn.DataParallel(model)
    model.eval()

    n_examples = 0

    # Iterate over data.
    avg_acc = []
    if data_name == 'DFAUST':
        pred_labels_per_video = [[] for _ in range(len(test_dataset.action_dataset.vertices))]
        logits_per_video = [[] for _ in range(len(test_dataset.action_dataset.vertices))]
    elif data_name == 'IKEA_EGO':
        pred_labels_per_video = [[] for _ in range(len(test_dataset.video_list))]
        logits_per_video = [[] for _ in range(len(test_dataset.video_list))]
    else:
        raise NotImplementedError
    for test_batchind, data in enumerate(test_dataloader):
        # get the inputs
        if data_name == 'DFAUST':
            inputs, labels_int, seq_idx, frame_pad = data['points'], data['labels'], data['seq_idx'], data['padding']
            inputs = inputs.permute(0, 1, 3, 2).cuda().requires_grad_().contiguous()
            labels = F.one_hot(labels_int.to(torch.int64), num_classes).permute(0, 2, 1).float().cuda()
        elif data_name == 'IKEA_EGO':
            inputs, labels, seq_idx, frame_pad = data
            # wrap them in Variable
            if not seq_idx[0] == seq_idx[1]:
                print("I am here")
            inputs = inputs.cuda().requires_grad_().contiguous()
            labels = labels.cuda()
            in_channel = cfg['DATA']['in_channel']
            inputs = inputs[:, :, 0:in_channel, :].contiguous()
        else:
            raise NotImplementedError
        out_dict = model(inputs)
        logits = out_dict['pred']

        acc = i3d_utils.accuracy_v2(torch.argmax(logits, dim=1), torch.argmax(labels, dim=1))
        avg_acc.append(acc.detach().cpu().numpy())
        n_examples += batch_size
        print('batch Acc: {}, [{} / {}]'.format(acc.item(), test_batchind, len(test_dataloader)))
        logits = logits.permute(0, 2, 1)
        logits = logits.reshape(inputs.shape[0] * frames_per_clip, -1)
        pred_labels = torch.argmax(logits, 1).detach().cpu().numpy()

        pred_labels_per_video, logits_per_video = \
            utils.accume_per_video_predictions(seq_idx, frame_pad, pred_labels_per_video, logits_per_video,
                                               pred_labels, logits, frames_per_clip)

    pred_labels_per_video = [np.array(pred_video_labels) for pred_video_labels in pred_labels_per_video]
    logits_per_video = [np.array(pred_video_logits) for pred_video_logits in logits_per_video]

    np.save(pred_output_filename, {'pred_labels': pred_labels_per_video, 'logits': logits_per_video})
    if data_name == 'DFAUST':
        utils.convert_frame_logits_to_segment_json(logits_per_video, json_output_filename,
                                                   test_dataset.action_dataset.sid_per_seq,
                                                   test_dataset.action_dataset.actions)
    elif data_name == 'IKEA_EGO':
        utils.convert_frame_logits_to_segment_json(logits_per_video, json_output_filename, test_dataset.video_list,
                                                   test_dataset.action_list)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    cfg = yaml.safe_load(open(os.path.join(args.logdir, args.identifier, 'config.yaml')))
    logdir = os.path.join(args.logdir, args.identifier)
    output_path = os.path.join(logdir, 'results')
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(logdir, args.model_ckpt)
    run(cfg, logdir, model_path, output_path)