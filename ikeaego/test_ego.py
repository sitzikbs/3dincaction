# Author: Yizhak Ben-Shabat (Itzik), 2023
# test on the ikea ASM dataset

import os
import sys
import yaml
sys.path.append('../')
import argparse
import i3d_utils
import sys
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from IKEAEgoDatasetClips import IKEAEgoDatasetClips as Dataset
import importlib.util
import utils as point_utils

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='./log/', help='path to model save dir')
parser.add_argument('--identifier', type=str, default='debug', help='unique run identifier')
parser.add_argument('--model_ckpt', type=str, default='000000.pt', help='checkpoint to load')
args = parser.parse_args()


def run(cfg, logdir, model_path, output_path):

    dataset_path = cfg['DATA']['dataset_path']
    pc_model = cfg['MODEL']['pc_model']
    batch_size = cfg['TESTING']['batch_size']
    in_channel = cfg['DATA']['in_channel']
    pred_output_filename = os.path.join(output_path, 'pred.npy')
    json_output_filename = os.path.join(output_path, 'action_segments.json')

    # setup dataset
    test_dataset = Dataset(dataset_path, set='test')

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                                  pin_memory=True)
    num_classes = test_dataset.num_classes
    frames_per_clip = test_dataset.frames_per_clip

    # setup the model
    checkpoints = torch.load(model_path)
    if pc_model == 'pn1':
        spec = importlib.util.spec_from_file_location("PointNet1", os.path.join(logdir, "pointnet.py"))
        pointnet = importlib.util.module_from_spec(spec)
        sys.modules["PointNet1"] = pointnet
        spec.loader.exec_module(pointnet)
        model = pointnet.PointNet1(k=num_classes, feature_transform=True, in_d=in_channel)
    elif pc_model == 'pn1_4d_basic':
        spec = importlib.util.spec_from_file_location("PointNet1Basic", os.path.join(logdir, "pointnet.py"))
        pointnet = importlib.util.module_from_spec(spec)
        sys.modules["PointNet1Basic"] = pointnet
        spec.loader.exec_module(pointnet)
        model = pointnet.PointNet1Basic(k=num_classes, feature_transform=True, n_frames=frames_per_clip,
                                        in_d=in_channel)
    elif pc_model == 'pn1_4d':
        spec = importlib.util.spec_from_file_location("PointNet4D", os.path.join(logdir, "pointnet.py"))
        pointnet = importlib.util.module_from_spec(spec)
        sys.modules["PointNet4D"] = pointnet
        spec.loader.exec_module(pointnet)
        model = pointnet.PointNet4D(k=num_classes, feature_transform=True, n_frames=frames_per_clip)
    elif pc_model == 'pn2':
            spec = importlib.util.spec_from_file_location("PointNet2",
                                                          os.path.join(logdir, "pointnet2_cls_ssg.py"))
            pointnet_pp = importlib.util.module_from_spec(spec)
            sys.modules["PointNet2"] = pointnet_pp
            spec.loader.exec_module(pointnet_pp)
            model = pointnet_pp.PointNet2(num_class=num_classes, n_frames=frames_per_clip, in_channel=in_channel)
    elif pc_model == 'pn2_4d':
            spec = importlib.util.spec_from_file_location("PointNetPP4D",
                                                          os.path.join(logdir, "pointnet2_cls_ssg.py"))
            pointnet_pp = importlib.util.module_from_spec(spec)
            sys.modules["PointNetPP4D"] = pointnet_pp
            spec.loader.exec_module(pointnet_pp)
            model = pointnet_pp.PointNetPP4D(num_class=num_classes, n_frames=frames_per_clip)
    elif pc_model == 'pn2_4d_basic':
            spec = importlib.util.spec_from_file_location("PointNet2Basic",
                                                          os.path.join(logdir, "pointnet2_cls_ssg.py"))
            pointnet_pp = importlib.util.module_from_spec(spec)
            sys.modules["PointNet2Basic"] = pointnet_pp
            spec.loader.exec_module(pointnet_pp)
            model = pointnet_pp.PointNet2Basic(num_class=num_classes, n_frames=frames_per_clip, in_channel=in_channel)
    elif pc_model == 'pn2_patchlets':
            spec = importlib.util.spec_from_file_location("PointNet2Patchlets",
                                                          os.path.join(logdir, "patchlets.py"))
            pointnet_pp = importlib.util.module_from_spec(spec)
            sys.modules["PointNet2Patchlets"] = pointnet_pp
            spec.loader.exec_module(pointnet_pp)
            model = pointnet_pp.PointNet2Patchlets(cfg=cfg['MODEL']['PATCHLET'], num_class=num_classes,
                                                      n_frames=frames_per_clip, in_channel=in_channel)
    elif pc_model == '3dmfv':
            spec = importlib.util.spec_from_file_location("FourDmFVNet",
                                                          os.path.join(logdir, "pytorch_3dmfv.py"))
            pytorch_3dmfv = importlib.util.module_from_spec(spec)
            sys.modules["FourDmFVNet"] = pytorch_3dmfv
            spec.loader.exec_module(pytorch_3dmfv)
            model = pytorch_3dmfv.FourDmFVNet(n_gaussians=cfg['MODEL']['3DMFV']['n_gaussians'], num_classes=num_classes,
                                              n_frames=frames_per_clip)

    model.load_state_dict(checkpoints["model_state_dict"])  # load trained model
    model.cuda()
    model = nn.DataParallel(model)

    n_examples = 0

    # Iterate over data.
    avg_acc = []
    pred_labels_per_video = [[] for i in range(len(test_dataset.video_list))]
    logits_per_video = [[] for i in range(len(test_dataset.video_list)) ]
    # last_vid_idx = 0
    for test_batchind, data in enumerate(test_dataloader):
        model.train(False)
        # get the inputs
        inputs, labels, vid_idx, frame_pad = data
        # wrap them in Variable
        if not vid_idx[0] == vid_idx[1]:
            print("I am here")
        inputs = inputs.cuda().requires_grad_().contiguous()
        labels = labels.cuda()

        inputs = inputs[:, :, 0:in_channel, :].contiguous()
        out_dict = model(inputs)
        logits = out_dict['pred']

        acc = i3d_utils.accuracy_v2(torch.argmax(logits, dim=1), torch.argmax(labels, dim=1))
        avg_acc.append(acc.item())
        n_examples += batch_size
        print('batch Acc: {}, [{} / {}]'.format(acc.item(), test_batchind, len(test_dataloader)))
        logits = logits.permute(0, 2, 1)
        logits = logits.reshape(inputs.shape[0] * frames_per_clip, -1)
        pred_labels = torch.argmax(logits, 1).detach().cpu().numpy()
        logits = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy().tolist()

        pred_labels_per_video, logits_per_video = \
            utils.accume_per_video_predictions(vid_idx, frame_pad, pred_labels_per_video, logits_per_video, pred_labels,
                                     logits, frames_per_clip)

    pred_labels_per_video = [np.array(pred_video_labels) for pred_video_labels in pred_labels_per_video]
    logits_per_video = [np.array(pred_video_logits) for pred_video_logits in logits_per_video]

    print("saving results to : " + pred_output_filename)
    np.save(pred_output_filename, {'pred_labels': pred_labels_per_video, 'logits': logits_per_video})
    utils.convert_frame_logits_to_segment_json(logits_per_video, json_output_filename, test_dataset.video_list,
                                               test_dataset.action_list)


if __name__ == '__main__':
    cfg = yaml.safe_load(open(os.path.join(args.logdir, args.identifier, 'config.yaml')))
    logdir = os.path.join(args.logdir, args.identifier)
    output_path = os.path.join(logdir, 'results')
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(logdir, args.model_ckpt)

    run(cfg, logdir, model_path, output_path)
