import torch
import visualization
import numpy as np
from DfaustDataset import DfaustActionClipsDataset
from ikeaaction.IKEAActionDataset import IKEAActionVideoClipDataset
import os
# import pandas as pd


from models.patchlets import PatchletsExtractor

dataset_name = 'ikea'
output_dir = './patchlet_parameter_ablations/'
n_examples = 1000

npoints = 512
k = 32
sample_mode = 'nn'
dfaust_augmentation = ['']
add_centroid_jitter = 0.001


def main():

    extract_pachlets = PatchletsExtractor(k=k, npoints=npoints, sample_mode=sample_mode,
                                          add_centroid_jitter=add_centroid_jitter)

    if dataset_name == 'ikea':
        dataset_path = '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/'
        dataset = IKEAActionVideoClipDataset(dataset_path, frames_per_clip=64, set='train', n_points=4096, input_type='pc', camera='dev3',
                          mode='img', cache_capacity=1)
    else:
        dataset_path = '/home/sitzikbs/Datasets/dfaust/'
        dataset = DfaustActionClipsDataset(dataset_path, frames_per_clip=64, set='train', n_points=1024,
                                shuffle_points='fps_each_frame', gender='female', data_augmentation=dfaust_augmentation )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=8,
                                                   pin_memory=True, shuffle=False, drop_last=True)

    degredation_rate = []
    total_degredation_percentage = []
    patchlet_variance = []
    example_counter = 0
    for batch_ind, data in enumerate(dataloader):

        if dataset_name == 'ikea':
            point_seq = data[0][..., :3, :].permute(0, 1, 3, 2).cuda()
        else:
            point_seq = data['points'].cuda()

        b, t, n, d = point_seq.shape
        patchlet_dict = extract_pachlets(point_seq)
        patch_idxs = patchlet_dict['patchlets']
        patchlet_points = patchlet_dict['patchlet_points']
        fps_idx = patchlet_dict['fps_idx']

        for batch_ind in range(b):
            example_counter = example_counter+1
            u_points = []

            for j in range(t):
                u_points.append(len(torch.unique(patchlet_points[batch_ind, j].reshape(-1, 3), dim=0)))

            u_points = np.array(u_points)
            shifted_u_points = np.concatenate([[n], u_points])
            point_diff = shifted_u_points[:-1] - u_points

            degredation_rate.append(np.mean(point_diff))
            total_degredation_percentage.append(100*(1 - u_points[-1] / n))
            patchlet_variance.append(torch.var(patchlet_points[batch_ind].permute(1, 0, 2, 3).reshape(npoints, -1, 3),
                                               dim=1).mean(0).norm().cpu().numpy())
        print('running average Degredation rate: {} points per frame'.format(np.mean(degredation_rate)))
        print('running average total degredation percentage (last - first): {}'.format(
            round(np.mean(total_degredation_percentage), 2)))
        print('running average of patchlet variance: {}'.format(np.mean(patchlet_variance)))
        if example_counter == n_examples:
            print("completed analizing {} examples".format(n_examples))
            break

        # visualization.pc_patchlet_points_vis(patchlet_dict['patchlet_points'][0].detach().cpu().numpy())
    final_degradation_rate = round(np.mean(total_degredation_percentage), 2)
    final_average_patchlet_variance = np.mean(patchlet_variance)
    print('Average Degredation rate: {} points per frame'.format(np.mean(degredation_rate)))
    print('Average total degredation percentage (last - first): {}'.format(final_degradation_rate))
    print('Average total variance: {}'.format(final_average_patchlet_variance))



if __name__ == "__main__":
    main()


