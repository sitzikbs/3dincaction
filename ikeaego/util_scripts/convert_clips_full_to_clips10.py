import torch
import os
import pickle
import numpy as np
import argparse
import json
import sys
sys.path.append('../')
from IKEAEgoDatasetClips import IKEAEgoDatasetClips as Dataset
import pandas as pd

def read_gt_json(ground_truth_filename, subset):
    """Reads ground truth file, checks if it is well formatted, and returns
       the ground truth instances and the activity classes.
    Parameters
    ----------
    ground_truth_filename : str
        Full path to the ground truth json file.
    Outputs
    -------
    ground_truth : df
        Data frame containing the ground truth instances.
    activity_index : dict
        Dictionary containing class index.
    """
    with open(ground_truth_filename, 'r') as fobj:
        data = json.load(fobj)
    # Checking format
    if not all([field in data.keys() for field in ['version', 'database'] ]):
        raise IOError('Please input a valid ground truth file.')

    # Read ground truth data.
    activity_index, cidx = {}, 0
    video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
    for videoid, v in data['database'].items():
        if not subset == v['subset']:
            continue
        for ann in v['annotation']:
            if ann['label'] not in activity_index:
                activity_index[ann['label']] = cidx
                cidx += 1
            video_lst.append(videoid)
            t_start_lst.append(float(ann['segment'][0]))
            t_end_lst.append(float(ann['segment'][1]))
            label_lst.append(activity_index[ann['label']])

    ground_truth = pd.DataFrame({'video-id': video_lst,
                                 't-start': t_start_lst,
                                 't-end': t_end_lst,
                                 'label': label_lst})
    return ground_truth, activity_index


def save_aux_data(out_path, subset, clip_set, clip_label_count, num_classes, video_list, frames_per_clip, action_labels, action_list):
    out_dict = {'clip_set': clip_set,
                'clip_label_count': clip_label_count,
                'num_classes': num_classes,
                'video_list': video_list,
                'action_list': action_list,
                'frames_per_clip':frames_per_clip,
                'action_labels': action_labels}

    with open(os.path.join(out_path, subset + '_aux.pickle'), 'wb') as f:
        pickle.dump(out_dict, f)

def write_gt_json(output_dataset_dir, data_df, action_list):
    json_filename = os.path.join(output_dataset_dir, 'gt_segments.json')
    final_dict = {'version': '2.0.1', 'database':{}}
    for subset in data_df:
        for i, row in data_df[subset].iterrows():
            segment_dict = {"segment": [row['t-start'], row['t-end']],
                               "label": action_list[row['label']]}
            if data_df[subset]['video-id'][i] not in final_dict['database']:
                final_dict['database'][row['video-id']] = {"subset": subset,  "annotation": [segment_dict]}
            else:
                final_dict['database'][row['video-id']]["annotation"].append(segment_dict)
    with open(json_filename, 'w') as fobj:
        json.dump(final_dict, fobj)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/home/sitzikbs/Datasets/ikeaego_small_clips_frameskip4/32/',
                    help='path to ikea asm dataset with point cloud data')
parser.add_argument('--output_dataset_dir', type=str, default='/home/sitzikbs/Datasets/ikeaego_small_clips_top10_frameskip4/32/',
                    help='path to the output directory where the new model will be saved')
args = parser.parse_args()

dataset_path = args.dataset_path
output_dataset_dir = args.output_dataset_dir

gt_json_path = os.path.join(dataset_path, 'gt_segments.json')
out_json_path = os.path.join(output_dataset_dir, 'gt_segments.json')
subsets = ['train', 'test']
k=10

accume_gt_dict = {}
for subset in subsets:
    outdir = os.path.join(output_dataset_dir, subset)
    os.makedirs(outdir, exist_ok=True)

    dataset = Dataset(dataset_path, set=subset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    action_list = dataset.action_list

    gt_data = read_gt_json(gt_json_path, subset+'ing')
    label_count = dataset.get_dataset_statistics()
    if subset == 'train': # get the 10 most prominent classes in the training set
        top10_labels = np.flip(np.argpartition(label_count, -k)[-k:]).tolist()
        new_action_list = np.take(action_list, top10_labels).tolist()
    new_gt_df = gt_data[0][np.isin(gt_data[0]['label'].to_numpy(), top10_labels)]

    new_gt_df['label'] = new_gt_df['label'].replace(top10_labels, np.arange(k))
    accume_gt_dict[subset+'ing'] = new_gt_df
    counter = 0
    clip_set = []
    clip_label_count = np.zeros(k)
    for batch_idx, data in enumerate(dataloader):
        inputs, labels, vid_idx, frame_pad = data
        label_ints = torch.argmax(labels.squeeze(), 0).cpu().numpy()
        if np.any(np.isin(label_ints, top10_labels[1:])) or subset=='test':
            new_labels = [top10_labels.index(label) if label in top10_labels else 0 for label in label_ints]
            new_labels_onehot = np.zeros((k, len(new_labels)))
            new_labels_onehot[new_labels, np.arange(len(new_labels))] = 1
            clip_label_count = clip_label_count + new_labels_onehot.sum(-1)
            out_dict = {'inputs': inputs.squeeze().detach().cpu().numpy(),
                        'labels': new_labels_onehot,
                        'vid_idx': vid_idx.squeeze().detach().cpu().numpy(),
                        'frame_pad': frame_pad.squeeze().detach().cpu().numpy()
                        }
            clip_set_data = list(dataset.clip_set[batch_idx])
            clip_set_data[1] = new_labels_onehot
            clip_set.append(tuple(clip_set_data))

            with open(os.path.join(outdir, str(counter). zfill(6)+'.pickle'), 'wb') as f:
                pickle.dump(out_dict, f)
            counter += 1
    new_action_labels = []
    for label_seq in dataset.action_labels:
        label_ints = np.argmax(label_seq, 1)
        new_labels = [top10_labels.index(label) if label in top10_labels else 0 for label in label_ints]
        new_labels_onehot = np.zeros(( len(new_labels), k))
        new_labels_onehot[np.arange(len(new_labels)), new_labels] = 1
        new_action_labels.append(new_labels_onehot)
    save_aux_data(output_dataset_dir, subset, clip_set,
                  clip_label_count, k, dataset.video_list, dataset.frames_per_clip, new_action_labels, new_action_list)

write_gt_json(output_dataset_dir, accume_gt_dict, new_action_list)