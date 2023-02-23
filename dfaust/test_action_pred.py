# Author: Yizhak Ben-Shabat (Itzik), 2022
# test 3DInAction on Dfaust dataset

import sys
sys.path.append('../')

import os
import argparse
import i3d_utils
import sys
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from DfaustDataset import DfaustActionClipsDataset as Dataset
import importlib.util
import visualization
import pathlib
import utils as point_utils
import yaml
from models.pointnet2_cls_ssg import PointNet2, PointNetPP4D, PointNet2Basic
from torch.multiprocessing import set_start_method

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='./log/', help='path to model save dir')
parser.add_argument('--identifier', type=str, default='debug', help='unique run identifier')
parser.add_argument('--model_ckpt', type=str, default='000000.pt', help='checkpoint to load')
args = parser.parse_args()

# from pointnet import PointNet4D
def run(cfg, logdir, model_path, output_path):

    dataset_path = cfg['DATA']['dataset_path']
    pc_model = cfg['MODEL']['pc_model']
    batch_size = cfg['TESTING']['batch_size']
    frames_per_clip = cfg['DATA']['frames_per_clip']
    n_points = cfg['DATA']['n_points']
    shuffle_points = cfg['DATA']['shuffle_points']
    gender = cfg['DATA']['gender']
    subset = cfg['TESTING']['set']
    pred_output_filename = os.path.join(output_path, subset + '_pred.npy')
    json_output_filename = os.path.join(output_path, subset + '_action_segments.json')
    aug = cfg['TESTING']['aug']
    noisy_data = cfg['DATA']['noisy_data']

    # setup dataset
    # test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    # test_transforms = transforms.Compose([transforms.CenterCrop(224)])

    test_dataset = Dataset(dataset_path, frames_per_clip=frames_per_clip, set=subset, n_points=n_points, last_op='pad',
                           data_augmentation=aug, shuffle_points=shuffle_points, gender=gender, noisy_data=noisy_data)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                                  pin_memory=True)
    num_classes = test_dataset.action_dataset.num_classes

    # setup the model
    checkpoints = torch.load(model_path)
    if pc_model == 'pn1':
        spec = importlib.util.spec_from_file_location("PointNet1", os.path.join(logdir, "pointnet.py"))
        pointnet = importlib.util.module_from_spec(spec)
        sys.modules["PointNet1"] = pointnet
        spec.loader.exec_module(pointnet)
        model = pointnet.PointNet1(k=num_classes, feature_transform=True)
    elif pc_model == 'pn1_4d_basic':
        spec = importlib.util.spec_from_file_location("PointNet1Basic", os.path.join(logdir, "pointnet.py"))
        pointnet = importlib.util.module_from_spec(spec)
        sys.modules["PointNet1Basic"] = pointnet
        spec.loader.exec_module(pointnet)
        model = pointnet.PointNet1Basic(k=num_classes, feature_transform=True, n_frames=frames_per_clip)
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
            model = pointnet_pp.PointNet2(num_class=num_classes, n_frames=frames_per_clip)
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
            model = pointnet_pp.PointNet2Basic(num_class=num_classes, n_frames=frames_per_clip)
    elif pc_model == 'pn2_patchlets':
            spec = importlib.util.spec_from_file_location("PointNet2Patchlets",
                                                          os.path.join(logdir, "patchlets.py"))
            pointnet_pp = importlib.util.module_from_spec(spec)
            sys.modules["PointNet2Patchlets"] = pointnet_pp
            spec.loader.exec_module(pointnet_pp)
            model = pointnet_pp.PointNet2Patchlets(cfg=cfg['MODEL']['PATCHLET'], num_class=num_classes, n_frames=frames_per_clip)
    elif pc_model == 'set_transformer':
            spec = importlib.util.spec_from_file_location("SetTransformerTemporal",
                                                          os.path.join(logdir, "set_transformer.py"))
            set_transformer = importlib.util.module_from_spec(spec)
            sys.modules["SetTransformerTemporal"] = set_transformer
            spec.loader.exec_module(set_transformer)
            model = set_transformer.SetTransformerTemporal(cfg=cfg['MODEL']['SET_TRANSFORMER'], num_classes=num_classes)
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


    pred_labels_per_video = [np.array(pred_video_labels) for pred_video_labels in pred_labels_per_video]
    logits_per_video = [np.array(pred_video_logits) for pred_video_logits in logits_per_video]

    np.save(pred_output_filename, {'pred_labels': pred_labels_per_video, 'logits': logits_per_video})
    utils.convert_frame_logits_to_segment_json(logits_per_video, json_output_filename, test_dataset.action_dataset.sid_per_seq,
                                               test_dataset.action_dataset.actions)


if __name__ == '__main__':
    cfg = yaml.safe_load(open(os.path.join(args.logdir, args.identifier, 'config.yaml')))
    logdir = os.path.join(args.logdir, args.identifier)
    output_path = os.path.join(logdir, 'results')
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(logdir, args.model_ckpt)
    run(cfg, logdir, model_path, output_path)


