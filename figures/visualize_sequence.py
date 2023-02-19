import torch
import sys
import os
import visualization
import numpy as np
import itertools
sys.path.append('../dfaust')
sys.path.append('../ikeaaction')
sys.path.append('../ikeaego')
from DfaustDataset import DfaustActionClipsDataset
from ikeaaction.IKEAActionDatasetClips import IKEAActionDatasetClips
from ikeaego.IKEAEgoDatasetClips import IKEAEgoDatasetClips
from models.patchlets import PatchletsExtractor
import time

def remove_patchlet_points_from_pc(point_seq, patchlet_point_list, features):
    '''
    get a point sequence and a patchlet point list sequence and remove the patchlet points from the sequence,
    used for visualization
    :param point_seq:
    :param patchlet_point_list:
    :return:
    '''
    t, n, _ = point_seq.shape
    out_points_seq = []
    out_feat_seq = []
    all_patchlet_points = np.concatenate(patchlet_point_list, axis=1)
    for i in range(t):
        rows_to_delete = []
        for j in range(n):
            # if np.any(np.linalg.norm(point_seq[i][j] - all_patchlet_points[i], axis=1) < 1e-2):
            if point_seq[i][j] in all_patchlet_points[i]:
                rows_to_delete.append(j)
        out_points_seq.append(np.delete(point_seq[i], rows_to_delete, 0))
        out_feat_seq.append(np.delete(features[i], rows_to_delete, 0))
    return out_points_seq, out_feat_seq


# def remove_patchlet_points_from_pc(point_seq, patchlet_ids):
#     out_points_seq = []
#     all_patchlet_idxs = np.concatenate(patchlet_ids, axis=1)
#     for i, pc in enumerate(point_seq):
#         idx = all_patchlet_idxs[i]
#         out_points_seq.append(np.delete(pc, idx, 0))
#     return out_points_seq

gender = 'female'
dataset_name = 'ikea'
# outdir = os.path.join('./log/sequence_images/', dataset_name, gender)
# os.makedirs(outdir, exist_ok=True)
view = 'iso'
show_patchlets, show_full_pc, reduce_opacity = True, True, False
# n_sequences = 1
sequence_id = [39, 50, 60, 10, 20, 30]
# patchlet_ids = [2, 50, 100]
patchlet_ids = [400, 151, 180]
frames_per_clip = 64
point_size = 15
k = 64
n_points = 1024
if dataset_name == 'ikea':
    dataset_path = os.path.join('/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller_clips/', str(frames_per_clip))
    dataset = IKEAActionDatasetClips(dataset_path, set='test')
elif dataset_name == 'dfaust':
    dataset_path = '/home/sitzikbs/Datasets/dfaust/'
    dataset = DfaustActionClipsDataset(dataset_path, frames_per_clip=frames_per_clip, set='test', n_points=n_points,
                                       shuffle_points='fps_each', gender=gender,
                                       noisy_data={'test': False, 'train':False})
elif dataset_name == 'ikeaego':
    dataset_path = os.path.join('/home/sitzikbs/Datasets/ikeaego_small_clips/',  str(frames_per_clip))
    dataset = IKEAEgoDatasetClips(dataset_path, set='test')
else:
    raise ValueError("unsupported dataset")


dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

extract_pachlets = PatchletsExtractor(k=k, npoints=512, sample_mode='nn',
                                      add_centroid_jitter=0.0, downsample_method='fps',
                                      radius=0.2)


for batch_ind, data in enumerate(dataloader):
    print("processing batch {}".format(batch_ind))
    if dataset_name == 'ikea' or dataset_name == 'ikeaego':
        point_seq = data[0][..., :3, :].permute(0, 1, 3, 2)
        point_color = data[0][..., 3:, :].permute(0, 1, 3, 2)/255
        labels_id = torch.argmax(data[1].squeeze(), 0)
        label_txt = [dataset.action_list[laebl_id.item()] for laebl_id in labels_id]
        if dataset_name == 'ikea':
            pc_color = data[0][..., 3:, :].permute(0, 1, 3, 2).squeeze().cpu().numpy() / 255
        elif dataset_name == 'ikeaego':
            pc_color = data[0][..., 6:, :].permute(0, 1, 3, 2).squeeze().cpu().numpy() / 255
    elif dataset_name == 'dfaust':
        point_seq = data['points']
        label_txt = None  # TODO add support for dfaust labels
        pc_color = None


    # if n_sequences == 0 or batch_ind < n_sequences:
    if batch_ind in sequence_id:
        start_t = time.time()
        patchlet_dict = extract_pachlets(point_seq.cuda())
        end_t = time.time()
        print("Computing patchlets took {} s".format(end_t - start_t))
        patchlet_points = [patchlet_dict['patchlet_points'].squeeze()[:, id].cpu().numpy() for id in patchlet_ids]


        visualization.pc_seq_vis(point_seq.squeeze().cpu().numpy(), text=label_txt, color=pc_color, point_size=15)




