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
import pyvista as pv
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./log/pn1_4d_f64_p1024_shuffle_each_aug1_b8_u2/',
                    help='path to model save dir')
parser.add_argument('--model', type=str, default='000200.pt', help='file name of model checkpint to load')
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/dfaust/', help='path to dataset')
parser.add_argument('--set', type=str, default='test', help='test | train set to evaluate ')
params = parser.parse_args()


args = torch.load(os.path.join(params.model_path, 'params.pth'))

output_path = os.path.join(params.model_path, 'result_visualization')
os.makedirs(output_path, exist_ok=True)
vid_output_path = os.path.join(params.model_path, 'result_visualization', 'vids')
os.makedirs(vid_output_path, exist_ok=True)
model_path = os.path.join(params.model_path, params.model)


test_dataset = Dataset(params.dataset_path, frames_per_clip=args.frames_per_clip, set=params.set, n_points=None,
                       last_op='none', shuffle_points='none')

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                              pin_memory=True)
num_classes = test_dataset.action_dataset.num_classes

#  setup the model
checkpoints = torch.load(model_path)
model = utils.get_model(args.pc_model, num_classes, args)
model.load_state_dict(checkpoints["model_state_dict"])  # load trained model
model.cuda()
model = nn.DataParallel(model)

faces = test_dataset.action_dataset.faces
faces = np.concatenate([3 * np.ones([faces.shape[0], 1], dtype=np.int16), faces], axis=1)
accume_grads = []
accume_text = []
accume_verts = []
accume_imgs = []
# Iterate over data.
for test_batchind, data in enumerate(test_dataloader):
    model.train(False)
    # get the inputs
    inputs, labels_int, seq_idx, subseq_pad = data['points'], data['labels'], data['seq_idx'], data['padding']
    if test_batchind == 0:
        last_seq_idx = seq_idx
    if not last_seq_idx == seq_idx or test_batchind == len(test_dataloader):
        frame_counter = 0
        seq_grads = np.concatenate(accume_grads, axis=1).squeeze()
        grad_mean, grad_std  = np.mean(seq_grads), np.std(seq_grads)
        seq_grads = np.clip(seq_grads, grad_mean - 2*grad_std, grad_mean + 2*grad_std)

        for i, frame_batch in enumerate(accume_verts):
            for j, frame in enumerate(frame_batch):
                mesh = pv.PolyData(frame, faces)
                mesh['scalars'] = seq_grads[i * args.frames_per_clip + j]
                pl = pv.Plotter(off_screen=True)
                pl.add_mesh(mesh, scalars=mesh['scalars'])
                pl.add_title(accume_text[i][j])

                pl.camera.position = (1, 1, 1)
                pl.camera.focal_point = (0, 0, 0)
                pl.camera.up = (0.0, 1.0, 0.0)
                pl.camera.zoom(0.5)
                img_filename = os.path.join(output_path, 'seq_{}_frame_{}.png'.format(int(last_seq_idx.cpu().numpy()),
                                                                                      frame_counter))
                pl.show(screenshot=img_filename)
                print("Saving " + img_filename + "... \n")
                accume_imgs.append(pl.image)
                frame_counter = frame_counter+1
                # pl.close()

        height, width, _ = accume_imgs[0].shape
        video = cv2.VideoWriter(os.path.join(vid_output_path, 'seq_{}.mp4'.format(int(last_seq_idx.detach().cpu().numpy()))),
                                cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        for image in accume_imgs:
            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.destroyAllWindows()
        video.release()

        accume_grads = []
        accume_text = []
        accume_verts = []
        accume_imgs = []
        last_seq_idx = seq_idx

    inputs = inputs.permute(0, 1, 3, 2).cuda().requires_grad_().contiguous()
    labels = F.one_hot(labels_int.to(torch.int64), num_classes).permute(0, 2, 1).float().cuda()

    out_dict = model(inputs)
    logits = out_dict['pred']

    gradients = utils.gradient(inputs, logits, create_graph=False, retain_graph=False).detach().clone()
    grad_mag = torch.linalg.norm(gradients, dim=2).detach().cpu().numpy()


    logits = logits.permute(0, 2, 1)
    logits = logits.reshape(inputs.shape[0] * inputs.shape[1], -1)
    pred_labels = torch.argmax(logits, 1).detach().cpu().numpy()


    vis_txt = ['GT = ' + test_dataset.action_dataset.actions[int(labels_int[0][j].detach().cpu().numpy())] + ', Pred = '
    + test_dataset.action_dataset.actions[int(pred_labels.reshape(-1, inputs.shape[1])[0][j])] for j in  range(inputs.shape[1])]

    last_seq_idx = seq_idx
    accume_verts.append(inputs[0].permute(0, 2, 1).detach().cpu().numpy())
    accume_text.append(vis_txt)
    accume_grads.append(grad_mag)



