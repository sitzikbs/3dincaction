# Author: Yizhak Ben-Shabat (Itzik), 2020
# test I3D on the ikea ASM dataset

import os
import sys
sys.path.append('../')
import argparse
import i3d_utils
import sys
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
from torchvision import transforms
# import videotransforms
import numpy as np
# from pytorch_i3d import InceptionI3d
from IKEAActionDataset import IKEAActionVideoClipDataset as Dataset
import importlib.util
import models.correformer as cf
from models.my_sinkhorn import SinkhornCorr
import utils as point_utils


parser = argparse.ArgumentParser()
parser.add_argument('--input_type', type=str, default='pc', help='rgb | depth, indicating which data to load')
parser.add_argument('--pc_model', type=str, default='pn1', help='which model to use for point cloud processing: pn1 | pn2 ')
parser.add_argument('--frame_skip', type=int, default=1, help='reduce fps by skipping frames')
parser.add_argument('--frames_per_clip', type=int, default=64, help='number of frames in a clip sequence')
parser.add_argument('--batch_size', type=int, default=2, help='number of clips per batch')
parser.add_argument('--n_points', type=int, default=2048, help='number of points in a point cloud')
parser.add_argument('--db_filename', type=str,
                    default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/ikea_annotation_db_full',
                    help='database file')
parser.add_argument('--model_path', type=str, default='./log/debug/',
                    help='path to model save dir')
parser.add_argument('--device', default='dev3', help='which camera to load')
parser.add_argument('--model', type=str, default='000000.pt', help='path to model save dir')
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/', help='path to dataset')
parser.add_argument('--use_pointlettes', type=int, default=0, help=' toggle to use pointlettes in the data loader'
                                                                   ' to sort the points temporally')
parser.add_argument('--pointlet_mode', type=str, default='none', help='choose pointlet creation mode kdtree | sinkhorn')
parser.add_argument('--n_gaussians', type=int, default=8, help='number of gaussians for 3DmFV representation')
parser.add_argument('--correformer', type=str, default='none',  help='None or path to correformer model')
parser.add_argument('--sort_model', type=str, default='sinkhorn', help='transformer | sinkhorn | none')

parser.add_argument('--patchlet_centroid_jitter', type=float, default=0.005,
                    help='jitter to add to nearest neighbor when generating the patchlets')
parser.add_argument('--patchlet_sample_mode', type=str, default='nn', help='nn | randn | mean type of patchlet sampling')
parser.add_argument('--k', type=int, default=16, help='number of nearest neighbors')
args = parser.parse_args()


# from pointnet import PointNet4D
def run(dataset_path, db_filename, model_path, output_path, frames_per_clip=64, input_type='rgb',
        testset_filename='test_cross_env.txt', trainset_filename='train_cross_env.txt', frame_skip=1,
        batch_size=8, device='dev3', n_points=None, pc_model='pn1', use_pointlettes=0):

    use_pointlettes = True if not args.use_pointlettes == 0 else False

    pred_output_filename = os.path.join(output_path, 'pred.npy')
    json_output_filename = os.path.join(output_path, 'action_segments.json')

    # setup dataset
    # test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    test_transforms = transforms.Compose([transforms.CenterCrop(224)])

    test_dataset = Dataset(dataset_path, db_filename=db_filename, test_filename=testset_filename,
                           train_filename=trainset_filename, transform=test_transforms, set='test', camera=device,
                           frame_skip=frame_skip, frames_per_clip=frames_per_clip, resize=None, mode='img',
                           input_type=input_type, n_points=n_points, use_pointlettes=use_pointlettes,
                           pointlet_mode=args.pointlet_mode
                           )

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                                  pin_memory=True)
    num_classes = test_dataset.num_classes
    # setup the model
    checkpoints = torch.load(model_path)
    if input_type == 'flow':
        spec = importlib.util.spec_from_file_location("InceptionI3d", os.path.join(args.model_path, "pytorch_i3d.py"))
        pytorch_i3d = importlib.util.module_from_spec(spec)
        sys.modules["pytorch_i3d"] = pytorch_i3d
        spec.loader.exec_module(pytorch_i3d)
        model = pytorch_i3d.InceptionI3d(400, in_channels=2)
        model.replace_logits(num_classes)
    elif input_type == 'rgb' or input_type == 'depth':
        spec = importlib.util.spec_from_file_location("InceptionI3d", os.path.join(args.model_path, "pytorch_i3d.py"))
        pytorch_i3d = importlib.util.module_from_spec(spec)
        sys.modules["pytorch_i3d"] = pytorch_i3d
        spec.loader.exec_module(pytorch_i3d)
        model = pytorch_i3d.InceptionI3d(157, in_channels=3)
        model.replace_logits(num_classes)
    elif input_type == 'pc':
        if pc_model == 'pn1':
            spec = importlib.util.spec_from_file_location("PointNet1", os.path.join(args.model_path, "pointnet.py"))
            pointnet = importlib.util.module_from_spec(spec)
            sys.modules["PointNet1"] = pointnet
            spec.loader.exec_module(pointnet)
            model = pointnet.PointNet1(k=num_classes, feature_transform=True)
        elif pc_model == 'pn1_4d_basic':
            spec = importlib.util.spec_from_file_location("PointNet1Basic", os.path.join(args.model_path, "pointnet.py"))
            pointnet = importlib.util.module_from_spec(spec)
            sys.modules["PointNet1Basic"] = pointnet
            spec.loader.exec_module(pointnet)
            model = pointnet.PointNet1Basic(k=num_classes, feature_transform=True, n_frames=frames_per_clip)
        elif pc_model == 'pn1_4d':
            spec = importlib.util.spec_from_file_location("PointNet4D", os.path.join(args.model_path, "pointnet.py"))
            pointnet = importlib.util.module_from_spec(spec)
            sys.modules["PointNet4D"] = pointnet
            spec.loader.exec_module(pointnet)
            model = pointnet.PointNet4D(k=num_classes, feature_transform=True, n_frames=frames_per_clip)
        elif pc_model == 'pn2':
                spec = importlib.util.spec_from_file_location("PointNetPP4D",
                                                              os.path.join(args.model_path, "pointnet2_cls_ssg.py"))
                pointnet_pp = importlib.util.module_from_spec(spec)
                sys.modules["PointNetPP4D"] = pointnet_pp
                spec.loader.exec_module(pointnet_pp)
                model = pointnet_pp.PointNetPP4D(num_class=num_classes, n_frames=frames_per_clip)
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
                                                          sample_mode = args.patchlet_sample_mode,
                                                          add_centroid_jitter = args.patchlet_centroid_jitter,
                                                          k=args.k )
        elif pc_model == '3dmfv':
                spec = importlib.util.spec_from_file_location("FourDmFVNet",
                                                              os.path.join(args.model_path, "pytorch_3dmfv.py"))
                pytorch_3dmfv = importlib.util.module_from_spec(spec)
                sys.modules["FourDmFVNet"] = pytorch_3dmfv
                spec.loader.exec_module(pytorch_3dmfv)
                model = pytorch_3dmfv.FourDmFVNet(n_gaussians=args.n_gaussians, num_classes=num_classes,
                                                  n_frames=frames_per_clip)
    else:
        raise ValueError("Unsupported input type")
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
    pred_labels_per_video = [[] for i in range(len(test_dataset.video_list))]
    logits_per_video = [[] for i in range(len(test_dataset.video_list)) ]
    # last_vid_idx = 0
    for test_batchind, data in enumerate(test_dataloader):
        model.train(False)
        # get the inputs
        inputs, labels, vid_idx, frame_pad = data
        if not args.sort_model == 'none':
            with torch.no_grad():
                inputs, _ = point_utils.sort_points(sort_model, inputs.permute(0, 1, 3, 2)[..., :3])
                inputs = inputs.permute(0, 1, 3, 2)
        # wrap them in Variable
        inputs = inputs.cuda().requires_grad_().contiguous()
        labels = labels.cuda()

        if input_type == 'pc':
            inputs = inputs[:, :, 0:3, :].contiguous()
            t = inputs.size(1)
            out_dict = model(inputs)
            logits = out_dict['pred']
            # logits = F.interpolate(logits, t, mode='linear', align_corners=True)
        else:
            t = inputs.size(2)
            logits = model(inputs)
            logits = F.interpolate(logits, t, mode='linear', align_corners=True)

        acc = i3d_utils.accuracy_v2(torch.argmax(logits, dim=1), torch.argmax(labels, dim=1))
        avg_acc.append(acc.item())
        n_examples += batch_size
        print('batch Acc: {}, [{} / {}]'.format(acc.item(), test_batchind, len(test_dataloader)))
        logits = logits.permute(0, 2, 1)
        logits = logits.reshape(inputs.shape[0] * frames_per_clip, -1)
        pred_labels = torch.argmax(logits, 1).detach().cpu().numpy()
        logits = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy().tolist()

        pred_labels_per_video, logits_per_video = \
            utils.accume_per_video_predictions(vid_idx, frame_pad,pred_labels_per_video, logits_per_video, pred_labels,
                                     logits, frames_per_clip)

    pred_labels_per_video = [np.array(pred_video_labels) for pred_video_labels in pred_labels_per_video]
    logits_per_video = [np.array(pred_video_logits) for pred_video_logits in logits_per_video]

    np.save(pred_output_filename, {'pred_labels': pred_labels_per_video, 'logits': logits_per_video})
    utils.convert_frame_logits_to_segment_json(logits_per_video, json_output_filename, test_dataset.video_list,
                                               test_dataset.action_list)


if __name__ == '__main__':
    # need to add argparse
    output_path = os.path.join(args.model_path, 'results')
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(args.model_path, args.model)
    run(dataset_path=args.dataset_path, db_filename=args.db_filename, model_path=model_path,
        output_path=output_path, frame_skip=args.frame_skip,  input_type=args.input_type, batch_size=args.batch_size,
        device=args.device, n_points=args.n_points, frames_per_clip=args.frames_per_clip, pc_model=args.pc_model,
        use_pointlettes=args.use_pointlettes)
    # os.system('python3 ../../evaluation/evaluate.py --results_path {} --dataset_path {} --mode vid'.format(output_path, args.dataset_path))
