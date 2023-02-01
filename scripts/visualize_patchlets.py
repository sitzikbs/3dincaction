import torch
import visualization
import numpy as np
from DfaustDataset import DfaustActionClipsDataset
from ikeaaction.IKEAActionDataset import IKEAActionVideoClipDataset
from torch.multiprocessing import set_start_method

from models.patchlets import PatchletsExtractor

if __name__ == "__main__":
    # set_start_method('spawn')

    dataset_name = 'ikea'
    npoints = 512
    k = 16
    sample_mode = 'nn'
    dfaust_augmentation = ['jitter']
    add_centroid_jitter = 0.005

    if dataset_name == 'ikea':
        dataset_path = '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/'
        dataset = IKEAActionVideoClipDataset(dataset_path, frames_per_clip=64, set='train', n_points=1024, input_type='pc', camera='dev3',
                          mode='img', cache_capacity=1)
    else:
        dataset_path = '/home/sitzikbs/Datasets/dfaust/'
        dataset = DfaustActionClipsDataset(dataset_path, frames_per_clip=64, set='train', n_points=1024,
                                shuffle_points='fps_each_frame', gender='female', data_augmentation=dfaust_augmentation )


    extract_pachlets = PatchletsExtractor(k=16, npoints=npoints, sample_mode=sample_mode, add_centroid_jitter=add_centroid_jitter)


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=8,
                                                   pin_memory=True, shuffle=False, drop_last=True)

    for batch_ind, data in enumerate(dataloader):

        if dataset_name == 'ikea':
            point_seq = data[0][..., :3, :].permute(0, 1, 3, 2).cuda()
        else:
            point_seq = data['points'].cuda()

        b, t, n, d = point_seq.shape
        patchlet_dict = extract_pachlets(point_seq)
        patch_idxs = patchlet_dict['patchlets']
        fps_idx = patchlet_dict['fps_idx']

        batch_ind = 2

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
