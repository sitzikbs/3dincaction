# Author: Yizhak Ben-Shabat (Itzik), 2022
# test 3DInAction on Dfaust dataset

#TODO export a full video sequence

import os
import argparse
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from DfaustDataset import DfaustActionClipsDataset as Dataset
import visualization
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./log/pn2_f64_p1024_shuffle_each_aug1_b8_u2/',
                    help='path to model save dir')
parser.add_argument('--model', type=str, default='000200.pt', help='file name of model checkpint to load')
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/dfaust/', help='path to dataset')
parser.add_argument('--set', type=str, default='test', help='test | train set to evaluate ')
params = parser.parse_args()

args = torch.load(os.path.join(params.model_path, 'params.pth'))

# output_path = os.path.join(params.model_path, 'result_visualization')
# os.makedirs(output_path, exist_ok=True)
model_path = os.path.join(params.model_path, params.model)


test_dataset = Dataset(params.dataset_path, frames_per_clip=args.frames_per_clip, set=params.set, n_points=None, last_op='pad',
                       shuffle_points='none')

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                              pin_memory=True)
num_classes = test_dataset.action_dataset.num_classes

#  setup the model
checkpoints = torch.load(model_path)
model = utils.get_model(args.pc_model, num_classes, args)
model.load_state_dict(checkpoints["model_state_dict"])  # load trained model
model.cuda()
model = nn.DataParallel(model)



# Iterate over data.
for test_batchind, data in enumerate(test_dataloader):
    model.train(False)
    # get the inputs
    inputs, labels_int, seq_idx, subseq_pad = data['points'], data['labels'], data['seq_idx'], data['padding']
    inputs = inputs.permute(0, 1, 3, 2).cuda().requires_grad_().contiguous()
    labels = F.one_hot(labels_int.to(torch.int64), num_classes).permute(0, 2, 1).float().cuda()

    out_dict = model(inputs)
    logits = out_dict['pred']

    gradients = utils.gradient(inputs, logits, create_graph=True, retain_graph=True).detach()
    grad_mag = torch.linalg.norm(gradients, dim=2).detach().cpu().numpy()
    grad_mean, grad_std = np.mean(grad_mag, axis=-1)[:, :, None], np.std(grad_mag, axis=-1)[:, :, None]
    grad_mag = np.clip(grad_mag, grad_mean - 2*grad_std, grad_mean + 2*grad_std)
    # acc = i3d_utils.accuracy_v2(torch.argmax(logits, dim=1), torch.argmax(labels, dim=1))

    # print('batch Acc: {}, [{} / {}]'.format(acc.item(), test_batchind, len(test_dataloader)))
    logits = logits.permute(0, 2, 1)
    logits = logits.reshape(inputs.shape[0] * args.frames_per_clip, -1)
    pred_labels = torch.argmax(logits, 1).detach().cpu().numpy()


    vis_txt = ['GT = ' + test_dataset.action_dataset.actions[int(labels_int[0][j].detach().cpu().numpy())] + ', Pred = '
    + test_dataset.action_dataset.actions[int(pred_labels.reshape(-1, args.frames_per_clip)[0][j])] for j in  range(args.frames_per_clip)]

    # visualization.pc_seq_vis(inputs[0].permute(0, 2,1).detach().cpu().numpy(), text=vis_txt, color=grad_mag[0])
    visualization.mesh_seq_vis(inputs[0].permute(0, 2, 1).detach().cpu().numpy(), test_dataset.action_dataset.faces,
                               text=vis_txt, color=grad_mag[0])



