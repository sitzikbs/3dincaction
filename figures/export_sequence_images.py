import torch
import sys
import os
import visualization
import numpy as np
import itertools
sys.path.append('../')
sys.path.append('../ikeaaction')
sys.path.append('../ikeaego')
sys.path.append('../dfaust')
from DfaustDataset import DfaustActionClipsDataset
from ikeaaction.IKEAActionDatasetClips import IKEAActionDatasetClips
from ikeaego.IKEAEgoDatasetClips import IKEAEgoDatasetClips
from models.patchlets import PatchletsExtractor
from scipy.spatial import cKDTree


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
        if features is not None:
            out_feat_seq.append(np.delete(features[i], rows_to_delete, 0))
    return out_points_seq, out_feat_seq

def get_point_density(points, k=30):
    tree = cKDTree(points)
    distances, idxs = tree.query(points, k, workers=-1)
    avg_distance = np.mean(distances, -1)
    density = 1. / avg_distance
    return density

# def remove_patchlet_points_from_pc(point_seq, patchlet_ids):
#     out_points_seq = []
#     all_patchlet_idxs = np.concatenate(patchlet_ids, axis=1)
#     for i, pc in enumerate(point_seq):
#         idx = all_patchlet_idxs[i]
#         out_points_seq.append(np.delete(pc, idx, 0))
#     return out_points_seq

gender = 'male'
dataset_name = 'dfaust'
outdir = os.path.join('./log/sequence_images/', dataset_name, gender)
os.makedirs(outdir, exist_ok=True)
view = 'front'
show_patchlets, show_full_pc, reduce_opacity = True, True, False

# n_sequences = 1
sequence_id = [99]
# patchlet_ids = [0, 1, 2]
patchlet_ids = [152, 600, 2500, 2300] #1850
frames_per_clip = 64

k = 64
n_points = 4096
point_size = 8 #[0.025 for _ in range(n_points)]
patchlet_point_size = 15
use_density_based_point_size = False  #if True patchlets coloring is not supported

if dataset_name == 'ikeaasm':
    dataset_path = '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller_clips/32/'
    dataset = IKEAActionDatasetClips(dataset_path, set='test')
elif dataset_name == 'ikeaego':
    dataset_path = '/home/sitzikbs/Datasets/ikeaego_small_clips_frameskip4/32/'
    dataset = IKEAEgoDatasetClips(dataset_path=dataset_path, set='test')
else:
    dataset_path = '/home/sitzikbs/Datasets/dfaust/'
    dataset = DfaustActionClipsDataset(dataset_path, frames_per_clip=frames_per_clip, set='test', n_points=n_points,
                                       shuffle_points='fps_each', gender=gender,
                                       noisy_data={'train': False, 'test': False})

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

extract_pachlets = PatchletsExtractor(k=k, npoints=n_points, sample_mode='nn',
                                      add_centroid_jitter=0.0, downsample_method='mean_var_t')

point_color = None

for batch_ind, data in enumerate(dataloader):
    with torch.no_grad():
        print("processing batch {}".format(batch_ind))
        if dataset_name == 'ikeaasm' or dataset_name == 'ikeaego':
            point_seq = data[0][..., :3, :].permute(0, 1, 3, 2)
            point_color = data[0][..., 3:6, :].permute(0, 1, 3, 2).squeeze().cpu().numpy()/255
        else:
            point_seq = data['points']

        # if n_sequences == 0 or batch_ind < n_sequences:
        if batch_ind in sequence_id:
            patchlet_dict = extract_pachlets(point_seq.cuda())

            # visualization.pc_patchlet_vis(point_seq.squeeze().cpu().numpy(), patchlet_dict['patchlet_points'].squeeze().cpu().numpy()) #use to find patchlet idxs

            patchlet_points = [patchlet_dict['patchlet_points'].squeeze()[:, id].cpu().numpy() for id in patchlet_ids]
            if show_patchlets == True:
                point_seq, point_color = remove_patchlet_points_from_pc(point_seq.squeeze().cpu().numpy(), patchlet_points,
                                                                        point_color)
                if dataset_name == 'dfaust':
                    point_color = None
            else:
                point_seq = point_seq.squeeze().cpu().numpy()

            # patchlet_idxs = [patchlet_dict['idx'].squeeze()[:, id].cpu().numpy() for id in patchlet_ids]
            # point_seq = remove_patchlet_points_from_pc(patchlet_dict['x_current'].squeeze().cpu().numpy(), patchlet_idxs)
            if use_density_based_point_size:
                point_size = (0.15*1/np.sqrt(get_point_density(point_seq[0])) ).tolist()
            visualization.export_pc_seq(point_seq, patchlet_points, text=None,
                                        color=point_color, cmap=None,
                                        point_size=point_size, output_path=os.path.join(outdir, str(batch_ind).zfill(6)),
                                        show_patchlets=show_patchlets, show_full_pc=show_full_pc,
                                        reduce_opacity=reduce_opacity, view=view)
            visualization.export_patchlet_seq(patchlet_points, point_size=patchlet_point_size,
                                              output_path=os.path.join(outdir, str(batch_ind).zfill(6)), view=view)
            visualization.export_patchlet_seq_separately(patchlet_points, point_size=patchlet_point_size,
                                              output_path=os.path.join(outdir, str(batch_ind).zfill(6)), view=view)




