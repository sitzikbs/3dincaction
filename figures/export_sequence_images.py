import torch
import sys
import os
import visualization
import numpy as np
import itertools
sys.path.append('../')
from DfaustDataset import DfaustActionClipsDataset
from ikeaaction.IKEAActionDatasetClips import IKEAActionDatasetClips
from models.patchlets import PatchletsExtractor

def remove_patchlet_points_from_pc(point_seq, patchlet_point_list):
    '''
    get a point sequence and a patchlet point list sequence and remove the patchlet points from the sequence,
    used for visualization
    :param point_seq:
    :param patchlet_point_list:
    :return:
    '''
    t, n, _ = point_seq.shape
    out_points_seq = []
    all_patchlet_points = np.concatenate(patchlet_point_list, axis=1)
    for i in range(t):
        rows_to_delete = []
        for j in range(n):
            # if np.any(np.linalg.norm(point_seq[i][j] - all_patchlet_points[i], axis=1) < 1e-2):
            if point_seq[i][j] in all_patchlet_points[i]:
                rows_to_delete.append(j)
        out_points_seq.append(np.delete(point_seq[i], rows_to_delete, 0))
    return out_points_seq


# def remove_patchlet_points_from_pc(point_seq, patchlet_ids):
#     out_points_seq = []
#     all_patchlet_idxs = np.concatenate(patchlet_ids, axis=1)
#     for i, pc in enumerate(point_seq):
#         idx = all_patchlet_idxs[i]
#         out_points_seq.append(np.delete(pc, idx, 0))
#     return out_points_seq


dataset_name = 'dfaust'
outdir = os.path.join('./log/sequence_images/', dataset_name)
os.makedirs(outdir, exist_ok=True)
view = 'front'
show_patchlets, show_full_pc, reduce_opacity = True, True, False
# n_sequences = 1
sequence_id = 14
patchlet_ids = [2, 50, 128]
frames_per_clip = 64


if dataset_name == 'ikea':
    dataset_path = '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/64/'
    dataset = IKEAActionDatasetClips(dataset_path, set='test')
else:
    dataset_path = '/home/sitzikbs/Datasets/dfaust/'
    dataset = DfaustActionClipsDataset(dataset_path, frames_per_clip=frames_per_clip, set='test', n_points=1024,
                                       shuffle_points='fps_each', gender='female')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

extract_pachlets = PatchletsExtractor(k=24, npoints=512, sample_mode='nn',
                                      add_centroid_jitter=0.0)


for batch_ind, data in enumerate(dataloader):
    print("processing batch {}".format(batch_ind))
    if dataset_name == 'ikea':
        point_seq = data[0][..., :3, :].permute(0, 1, 3, 2)
    else:
        point_seq = data['points']



    # if n_sequences == 0 or batch_ind < n_sequences:
    if sequence_id == batch_ind:
        patchlet_dict = extract_pachlets(point_seq.cuda())
        patchlet_points = [patchlet_dict['patchlet_points'].squeeze()[:, id].cpu().numpy() for id in patchlet_ids]
        point_seq = remove_patchlet_points_from_pc(point_seq.squeeze().cpu().numpy(), patchlet_points)
        # patchlet_idxs = [patchlet_dict['idx'].squeeze()[:, id].cpu().numpy() for id in patchlet_ids]
        # point_seq = remove_patchlet_points_from_pc(patchlet_dict['x_current'].squeeze().cpu().numpy(), patchlet_idxs)

        visualization.export_pc_seq(point_seq, patchlet_points, text=None,
                                    color=None, cmap=None,
                                    point_size=15, output_path=os.path.join(outdir, str(batch_ind).zfill(6)),
                                    show_patchlets=show_patchlets, show_full_pc=show_full_pc,
                                    reduce_opacity=reduce_opacity, view=view)




