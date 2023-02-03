import i3d_utils as utils
import torch
import os
import sys
import pickle
sys.path.append('../')
from IKEAActionDataset import IKEAActionVideoClipDataset as Dataset

dataset_path = '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/'
output_dataset_dir = '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller_clips/'
frames_per_clip = 64
set = 'train'
path = os.path.normpath(output_dataset_dir)
split_path = path.split(os.sep)
split_path[-1] = split_path[-1]+'_'+str(frames_per_clip)
split_path[0] ='/'
output_dataset_dir = os.path.join(*split_path)
n_points = 4096

outdir = os.path.join(output_dataset_dir, set)
os.makedirs(outdir, exist_ok=True)

dataset = Dataset(dataset_path, db_filename='ikea_annotation_db_full', train_filename='train_cross_env.txt',
                        test_filename='test_cross_env.txt', set=set, camera='dev3', frame_skip=1,
                        frames_per_clip=frames_per_clip, resize=None, mode='img', input_type='pc',
                        n_points=n_points, cache_capacity=1)
print("Number of clips in the dataset:{}".format(len(dataset)))
weights = utils.make_weights_for_balanced_classes(dataset.clip_set, dataset.clip_label_count)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)


out_dict = {'weights': weights,
            'clip_set': dataset.clip_set,
            'clip_label_count': dataset.clip_label_count}
with open(os.path.join(output_dataset_dir, set+'_aux.pickle'), 'wb') as f:
    pickle.dump(out_dict, f)

for train_batchind, data in enumerate(dataloader):
    inputs, labels, vid_idx, frame_pad = data
    inputs, labels, vid_idx, frame_pad = inputs.squeeze(), labels.squeeze(), vid_idx.squeeze(), frame_pad.squeeze()
    out_dict = {'inputs': inputs.detach().cpu().numpy(),
                'labels': labels.detach().cpu().numpy(),
                'vid_idx': vid_idx.detach().cpu().numpy(),
                'frame_pad': frame_pad.detach().cpu().numpy()
                }
    with open(os.path.join(outdir, str(train_batchind). zfill(6)+'.pickle'), 'wb') as f:
        pickle.dump(out_dict, f)