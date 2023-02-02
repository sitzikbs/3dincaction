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
from torchvision import transforms
import numpy as np
from DfaustDataset import DfaustActionClipsDataset as Dataset
import importlib.util
import visualization
import pathlib
import models.correformer as cf
import utils as point_utils
from models.my_sinkhorn import SinkhornCorr
from models.pointnet2_cls_ssg import PointNet2, PointNetPP4D, PointNet2Basic
from torch.multiprocessing import set_start_method

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--pc_model', type=str, default='pn1_4d', help='which model to use for point cloud processing: pn1 | pn2 ')
parser.add_argument('--frames_per_clip', type=int, default=64, help='number of frames in a clip sequence')
parser.add_argument('--batch_size', type=int, default=2, help='number of clips per batch')
parser.add_argument('--n_points', type=int, default=1024, help='number of points in a point cloud')
parser.add_argument('--model_path', type=str, default='./log/baseline_fps/dfaust_all_pn1_4d_f64_p1024_shuffle_fps_each_frame_aug1_b16_u1_sort.sinkhorn',
                    help='path to model save dir')
parser.add_argument('--model', type=str, default='000200.pt', help='path to model save dir')
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/dfaust/', help='path to dataset')
parser.add_argument('--n_gaussians', type=int, default=8, help='number of gaussians for 3DmFV representation')
parser.add_argument('--set', type=str, default='test', help='test | train set to evaluate ')
parser.add_argument('--shuffle_points', type=str, default='fps_each_frame', help='fps e each | none shuffle the input points '
                                                                       'at initialization | for each batch example | no shufll')
parser.add_argument('--visualize_results', type=int, default=False, help='visualzies the first subsequence in each batch')
parser.add_argument('--correformer', type=str,
                    default='none',
                    help='None or path to correformer model')
parser.add_argument('--gender', type=str,
                    default='female', help='female | male | all indicating which subset of the dataset to use')
parser.add_argument('--sort_model', type=str, default='sinkhorn', help='transformer | sinkhorn | none')

parser.add_argument('--patchlet_centroid_jitter', type=float, default=0.005,
                    help='jitter to add to nearest neighbor when generating the patchlets')
parser.add_argument('--patchlet_sample_mode', type=str, default='nn', help='nn | randn | mean type of patchlet sampling')
args = parser.parse_args()


# from pointnet import PointNet4D
def run(dataset_path, model_path, output_path, frames_per_clip=64,  batch_size=8, n_points=None, pc_model='pn1'):

    pred_output_filename = os.path.join(output_path, args.set + '_pred.npy')
    json_output_filename = os.path.join(output_path, args.set + '_action_segments.json')

    # setup dataset
    # test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    # test_transforms = transforms.Compose([transforms.CenterCrop(224)])

    test_dataset = Dataset(dataset_path, frames_per_clip=frames_per_clip, set=args.set, n_points=n_points, last_op='pad',
                           shuffle_points=args.shuffle_points, gender=args.gender)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                                                  pin_memory=True)
    num_classes = test_dataset.action_dataset.num_classes

    # setup the model
    checkpoints = torch.load(model_path)

    if pc_model == 'pn1':
            spec = importlib.util.spec_from_file_location("PointNet1", os.path.join(args.model_path, "pointnet.py"))
            pointnet = importlib.util.module_from_spec(spec)
            sys.modules["PointNet1"] = pointnet
            spec.loader.exec_module(pointnet)
            model = pointnet.PointNet1(k=num_classes, feature_transform=True)
    elif pc_model == 'pn1_4d':
            spec = importlib.util.spec_from_file_location("PointNet4D", os.path.join(args.model_path, "pointnet.py"))
            pointnet = importlib.util.module_from_spec(spec)
            sys.modules["PointNet4D"] = pointnet
            spec.loader.exec_module(pointnet)
            model = pointnet.PointNet4D(k=num_classes, feature_transform=True, n_frames=frames_per_clip)
    elif pc_model == 'pn1_4d_basic':
            spec = importlib.util.spec_from_file_location("PointNet1Basic", os.path.join(args.model_path, "pointnet.py"))
            pointnet = importlib.util.module_from_spec(spec)
            sys.modules["PointNet1Basic"] = pointnet
            spec.loader.exec_module(pointnet)
            model = pointnet.PointNet1Basic(k=num_classes, feature_transform=True, n_frames=frames_per_clip)
    elif pc_model == 'pn2':
            spec = importlib.util.spec_from_file_location("PointNet2",
                                                          os.path.join(args.model_path, "pointnet2_cls_ssg.py"))
            pointnet_pp = importlib.util.module_from_spec(spec)
            sys.modules["PointNet2"] = pointnet_pp
            spec.loader.exec_module(pointnet_pp)
            model = pointnet_pp.PointNet2(num_class=num_classes, n_frames=frames_per_clip)
    elif pc_model == 'pn2_4d':
            spec = importlib.util.spec_from_file_location("PointNetPP4D",
                                                          os.path.join(args.model_path, "pointnet2_cls_ssg.py"))
            pointnet_pp = importlib.util.module_from_spec(spec)
            sys.modules["PointNetPP4D"] = pointnet_pp
            spec.loader.exec_module(pointnet_pp)
            model = pointnet_pp.PointNetPP4D(num_class=num_classes, n_frames=frames_per_clip)
    elif pc_model == 'pn2_4d_basic':
        spec = importlib.util.spec_from_file_location("PointNet2Basic",
                                                      os.path.join(args.model_path, "pointnet2_cls_ssg.py"))
        pointnet_pp = importlib.util.module_from_spec(spec)
        sys.modules["PointNet2Basic"] = pointnet_pp
        spec.loader.exec_module(pointnet_pp)
        model = pointnet_pp.PointNet2Basic(num_class=num_classes, n_frames=frames_per_clip)
    elif pc_model == 'pn2_patchlets':
        spec = importlib.util.spec_from_file_location("PointNet2Patchlets_v2",
                                                      os.path.join(args.model_path, "patchlets.py"))
        pointnet_pp = importlib.util.module_from_spec(spec)
        sys.modules["PointNet2Patchlets_v2"] = pointnet_pp
        spec.loader.exec_module(pointnet_pp)
        model = pointnet_pp.PointNet2Patchlets_v2(num_class=num_classes, n_frames=frames_per_clip,
                                                  sample_mode=args.patchlet_sample_mode,
                                                  add_centroid_jitter=args.patchlet_centroid_jitter
                                                  )
    elif pc_model == '3dmfv':
            spec = importlib.util.spec_from_file_location("FourDmFVNet",
                                                          os.path.join(args.model_path, "pytorch_3dmfv.py"))
            pytorch_3dmfv = importlib.util.module_from_spec(spec)
            sys.modules["FourDmFVNet"] = pytorch_3dmfv
            spec.loader.exec_module(pytorch_3dmfv)
            model = pytorch_3dmfv.FourDmFVNet(n_gaussians=args.n_gaussians, num_classes=num_classes,
                                              n_frames=frames_per_clip)

    model.load_state_dict(checkpoints["model_state_dict"])  # load trained model
    model.cuda()
    model = nn.DataParallel(model)

    # Load correspondance transformer
    if args.sort_model == 'correformer':
        sort_model = cf.get_correformer(args.correformer)
    elif args.sort_model == 'sinkhorn':
        sort_model = SinkhornCorr(max_iters=10).cuda()

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
        if not args.sort_model == 'none':
            with torch.no_grad():
                inputs, _ = point_utils.sort_points(sort_model, inputs)
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


if __name__ == '__main__':
    # need to add argparse
    # set_start_method('spawn')
    output_path = pathlib.Path(os.path.join(args.model_path, 'results_'+str(args.model))).with_suffix("")
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(args.model_path, args.model)
    run(dataset_path=args.dataset_path,  model_path=model_path, output_path=output_path, batch_size=args.batch_size,
        n_points=args.n_points, frames_per_clip=args.frames_per_clip, pc_model=args.pc_model)
    # os.system('python3 ../../evaluation/evaluate.py --results_path {} --dataset_path {} --mode vid'.format(output_path, args.dataset_path))
