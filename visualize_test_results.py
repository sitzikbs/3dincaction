# Author: Yizhak Ben-Shabat (Itzik), 2022
# test 3DInAction on Dfaust dataset

import os
import argparse
import i3d_utils
import sys
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DfaustDataset import DfaustActionClipsDataset as Dataset
import importlib.util
import visualization


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./log/pn1_f1_p1024_shuffle_once_sampler_weighted/',
                    help='path to model save dir')
parser.add_argument('--model', type=str, default='000030.pt', help='path to model save dir')
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/dfaust/', help='path to dataset')
parser.add_argument('--set', type=str, default='test', help='test | train set to evaluate ')
parser.add_argument('--visualize_results', type=int, default=False, help='visualzies the first subsequence in each batch')
params = parser.parse_args()

args = torch.load(os.path.join(params.model_path, 'params.pth'))

output_path = os.path.join(args.model_path, 'result_visualization')
os.makedirs(output_path, exist_ok=True)
model_path = os.path.join(params.model_path, args.model)


test_dataset = Dataset(dataset_path, frames_per_clip=frames_per_clip, set=args.set, n_points=n_points, last_op='pad',
                       shuffle_points=args.shuffle_points)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                              pin_memory=True)
num_classes = test_dataset.action_dataset.num_classes

# setup the model
checkpoints = torch.load(model_path)

model = utils.get_model(args.pc_model, num_classes, args)
model.load_state_dict(checkpoints["model_state_dict"])  # load trained model
model.cuda()
model = nn.DataParallel(model)

n_examples = 0
# Iterate over data.
avg_acc = []
pred_labels_per_video = [[] for i in range(len(test_dataset.action_dataset.vertices))]
logits_per_video = [[] for i in range(len(test_dataset.action_dataset.vertices)) ]
# last_vid_idx = 0
for test_batchind, data in enumerate(test_dataloader):
    model.train(False)
    # get the inputs
    inputs, labels_int, seq_idx, subseq_pad = data['points'], data['labels'], data['seq_idx'], data['padding']
    inputs = inputs.permute(0, 1, 3, 2).cuda().requires_grad_().contiguous()
    labels = F.one_hot(labels_int.to(torch.int64), num_classes).permute(0, 2, 1).float().cuda()

    # inputs = inputs[:, :, 0:3, :].contiguous()
    # t = inputs.size(1)
    out_dict = model(inputs)
    logits = out_dict['pred']
    # logits = F.interpolate(logits, t, mode='linear', align_corners=True)


    acc = i3d_utils.accuracy_v2(torch.argmax(logits, dim=1), torch.argmax(labels, dim=1))
    avg_acc.append(acc.detach().cpu().numpy())
    n_examples += batch_size
    print('batch Acc: {}, [{} / {}]'.format(acc.item(), test_batchind, len(test_dataloader)))
    logits = logits.permute(0, 2, 1)
    logits = logits.reshape(inputs.shape[0] * frames_per_clip, -1)
    pred_labels = torch.argmax(logits, 1).detach().cpu().numpy()
    # logits = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy().tolist()

    pred_labels_per_video, logits_per_video = \
        utils.accume_per_video_predictions(seq_idx, subseq_pad, pred_labels_per_video, logits_per_video,
                                           pred_labels, logits, frames_per_clip)

    if args.visualize_results:
        vis_txt = ['GT = ' + test_dataset.action_dataset.actions[int(labels_int[0][j].detach().cpu().numpy())] + ', Pred = '
        + test_dataset.action_dataset.actions[int(pred_labels.reshape(-1, args.frames_per_clip)[0][j])] for j in  range(args.frames_per_clip)]
        visualization.pc_seq_vis(inputs[0].permute(0, 2,1).detach().cpu().numpy(), text=vis_txt)

pred_labels_per_video = [np.array(pred_video_labels) for pred_video_labels in pred_labels_per_video]
logits_per_video = [np.array(pred_video_logits) for pred_video_logits in logits_per_video]

np.save(pred_output_filename, {'pred_labels': pred_labels_per_video, 'logits': logits_per_video})
utils.convert_frame_logits_to_segment_json(logits_per_video, json_output_filename, test_dataset.action_dataset.sid_per_seq,
                                           test_dataset.action_dataset.actions)
