import torch
import visualization
import numpy as np
import sys
sys.path.append('../dfaust')
sys.path.append('../ikeaaction')
sys.path.append('../ikeaego')
from DfaustDataset import DfaustActionClipsDataset
from ikeaaction.IKEAActionDatasetClips import IKEAActionDatasetClips
from IKEAEgoDatasetClips import IKEAEgoDatasetClips
from torch.multiprocessing import set_start_method

from models.patchlets import PatchletsExtractor, PatchletsExtractorBidirectional

if __name__ == "__main__":
    # set_start_method('spawn')

    dataset_name = 'ikeaego'
    downsample_method = 'fps'
    npoints = 512
    k = 16
    sample_mode = 'nn'
    dfaust_augmentation = ['']
    add_centroid_jitter = 0.00
    bidirectional_extractor = True
    if dataset_name == 'ikea':
        dataset_path = '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller_clips/64/'
        dataset = IKEAActionDatasetClips(dataset_path,  set='train')
    elif dataset_name == 'ikeaego':
        dataset_path = '/home/sitzikbs/Datasets/ikeaego_small_clips/32/'
        dataset = IKEAEgoDatasetClips(dataset_path,  set='test')
    else:
        dataset_path = '/home/sitzikbs/Datasets/dfaust/'
        dataset = DfaustActionClipsDataset(dataset_path, frames_per_clip=64, set='train', n_points=1024,
                                shuffle_points='fps_each', gender='female',
                                           data_augmentation=dfaust_augmentation,
                                           noisy_data={'test': False, 'train': False})

    if bidirectional_extractor:
        extract_pachlets = PatchletsExtractorBidirectional(k=k, npoints=npoints, sample_mode=sample_mode,
                                                           add_centroid_jitter=add_centroid_jitter,
                                                           downsample_method=downsample_method,
                                                           radius=0.2)
    else:
        extract_pachlets = PatchletsExtractor(k=k, npoints=npoints, sample_mode=sample_mode,
                                              add_centroid_jitter=add_centroid_jitter, downsample_method=downsample_method,
                                              radius=0.2)



    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0,
                                                   pin_memory=True, shuffle=False, drop_last=True)

    for batch_ind, data in enumerate(dataloader):

        if dataset_name == 'ikea' or dataset_name == 'ikeaego':
            point_seq = data[0][..., :3, :].permute(0, 1, 3, 2).cuda()
        else:
            point_seq = data['points'].cuda()

        b, t, n, d = point_seq.shape
        patchlet_dict = extract_pachlets(point_seq)
        patch_idxs = patchlet_dict['patchlets']
        fps_idx = patchlet_dict['fps_idx']

        batch_ind = 0

        u_points = []
        for j in range(t):
            u_points.append(len(torch.unique(patchlet_dict['patchlet_points'][batch_ind, j].reshape(-1, 3), dim=0)))
        u_points = np.array(u_points)

        print('Maximum difference: ' + str(u_points[0] - u_points[-1]))

        shifted_u_points = u_points.copy()
        shifted_u_points = np.roll(shifted_u_points, -1)
        degredation_rate = np.mean(u_points[:-1] - shifted_u_points[:-1])
        print('degredation rate: ' + str(degredation_rate))
        visualization.pc_patchlet_points_vis(patchlet_dict['patchlet_points'][batch_ind].detach().cpu().numpy())
        # visualization.pc_patchlet_vis(point_seq[batch_ind].cpu().numpy(), patchlet_dict['patchlet_points'][batch_ind].cpu().numpy())
        # visualization.pc_patchlet_vis(patchlet_dict['patchlet_points'][batch_ind, :, :, 0].cpu().numpy(), patch_mask.astype(np.float32))
        # visualization.pc_patchlet_patch_vis(patchlet_dict['patchlet_points'][batch_ind].cpu().numpy(), patchlet_dict['distances'][batch_ind].cpu().numpy())
