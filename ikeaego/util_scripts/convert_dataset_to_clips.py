import sys
sys.path.append('../')
import i3d_utils as utils
import torch
import os
import pickle
import argparse
from EgoSmallRawDataset import HololensStreamRecClipDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/home/sitzikbs/Datasets/ikeaego_small/',
                    help='path to ikea asm dataset with point cloud data')
parser.add_argument('--output_dataset_dir', type=str, default='/home/sitzikbs/Datasets/ikeaego_small_clips_frameskip3/',
                    help='path to the output directory where the new model will be saved')
parser.add_argument('--frames_per_clip', type=int, default=32,  help='number of frames in each clip')
parser.add_argument('--frame_skip', type=int, default=3,  help='number frames to skip between consecutive frames')
args = parser.parse_args()

dataset_path = args.dataset_path
output_dataset_dir = args.output_dataset_dir
frames_per_clip = args.frames_per_clip


subsets = ['train', 'test']
# n_points = 4096
# for set in sets:
for subset in subsets:

    output_dataset_dir_w_frames = os.path.join(output_dataset_dir, str(frames_per_clip))
    outdir = os.path.join(output_dataset_dir_w_frames, subset)
    os.makedirs(outdir, exist_ok=True)

    dataset = HololensStreamRecClipDataset(dataset_path, subset=subset, frames_per_clip=frames_per_clip,
                                           modalities=['point_clouds'], frame_skip=args.frame_skip)
    gt_json_path = os.path.join(dataset_path, 'indexing_files', 'db_gt_annotations.json')
    os.system('cp %s %s' % (gt_json_path, os.path.join(output_dataset_dir_w_frames, 'gt_segments.json')))  # copy gt file

    print("Number of clips in the dataset:{}".format(len(dataset)))

    weights = utils.make_weights_for_balanced_classes(dataset.clip_set, dataset.clip_label_count)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)
    print("Saving Auxilary file...")
    out_dict = {'weights': weights,
                'clip_set': dataset.clip_set,
                'clip_label_count': dataset.clip_label_count,
                'num_classes': dataset.num_classes,
                'video_list': dataset.video_list,
                'action_list': dataset.action_list,
                'frames_per_clip': frames_per_clip,
                'action_labels': dataset.action_labels}

    with open(os.path.join(output_dataset_dir_w_frames, subset + '_aux.pickle'), 'wb') as f:
        pickle.dump(out_dict, f)

    print("Saving point cloud sequences file...")
    for train_batchind, data in enumerate(dataloader):
        inputs, labels, vid_idx, frame_pad = data
        inputs, labels, vid_idx, frame_pad = inputs['point_clouds'].squeeze(), labels.squeeze(), vid_idx.squeeze(), frame_pad.squeeze()
        out_dict = {'inputs': inputs.detach().cpu().numpy(),
                    'labels': labels.detach().cpu().numpy(),
                    'vid_idx': vid_idx.detach().cpu().numpy(),
                    'frame_pad': frame_pad.detach().cpu().numpy()
                    }

        with open(os.path.join(outdir, str(train_batchind). zfill(6)+'.pickle'), 'wb') as f:
            pickle.dump(out_dict, f)