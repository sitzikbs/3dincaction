import torch
import visualization
import numpy as np
import sys
sys.path.append('../dfaust/')
from DfaustDataset import DfaustActionClipsDataset
from ikeaaction.IKEAActionDataset import IKEAActionDatasetClips
import os
# import pandas as pd
import wandb
from models.patchlets import PatchletsExtractor
import matplotlib.pyplot as plt

sweep = False
dataset_name = 'ikea'
output_dir = './patchlet_parameter_ablations/'
n_examples = 1000
npoints = 512
frames_per_clip = 64

# Define sweep config

if sweep:
    k_list = [8, 16, 32, 64]
    sample_mode_list = ['nn']
    dfaust_augmentation = ['']
    centroid_noise_list = [0.1]
    # k_list = [8, 16, 32, 64]
    # sample_mode_list = ['nn', 'randn']

    # centroid_noise_list = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01]
    configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'final_degradation_rate'},
        'parameters':
        {
            'k': {'values': k_list},
            'sample_mode': {'values': sample_mode_list},
            'centroid_noise': {'values': centroid_noise_list}
         }
    }
    sweep_id = wandb.sweep(sweep=configuration, project='point_drop_sweep')
else:
    dfaust_augmentation = ['']
    configuration = {
            'k':  16,
            'sample_mode': 'nn',
            'centroid_noise':  0.001
    }



def main():

    k = wandb.config.k
    sample_mode = wandb.config.sample_mode
    add_centroid_jitter = wandb.config.centroid_noise

    extract_pachlets = PatchletsExtractor(k=k, npoints=npoints, sample_mode=sample_mode,
                                          add_centroid_jitter=add_centroid_jitter)

    if dataset_name == 'ikea':
        dataset_path = '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/'
        dataset = IKEAActionDatasetClips(dataset_path,  set='train')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=8,
                                                       pin_memory=True, shuffle=False, drop_last=True)
    else:
        dataset_path = '/home/sitzikbs/Datasets/dfaust/'
        dataset = DfaustActionClipsDataset(dataset_path, frames_per_clip=frames_per_clip, set='train', n_points=1024,
                                shuffle_points='fps_each_frame', gender='female', data_augmentation=dfaust_augmentation )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0,
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
            patchlet_var = []

            for j in range(t):
                u_points.append(len(torch.unique(patchlet_points[batch_ind, j].reshape(-1, 3), dim=0)))

            u_points = np.array(u_points)
            shifted_u_points = np.concatenate([[n], u_points])
            point_diff = shifted_u_points[:-1] - u_points

            degredation_rate.append(np.mean(point_diff))
            total_degredation_percentage.append(100*(1 - u_points[-1] / n))
            # patchlet_variance.append(torch.var(patchlet_points[batch_ind].permute(1, 0, 2, 3).reshape(npoints, -1, 3),
            #                                    dim=1).mean(0).norm().cpu().numpy())
            patchlet_variance.append(torch.var(patchlet_points[batch_ind].mean(-2), dim=0).norm(dim=-1).cpu().numpy())
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

    # plot histogram
    patchlet_variance_flattenned = np.vstack(patchlet_variance).reshape(-1)
    patchlet_variance_scores = [s for s in patchlet_variance_flattenned]
    fig = plt.figure()
    plt.hist(patchlet_variance_scores, bins=1000, range=[0, np.percentile(patchlet_variance_scores, 90)])
    fig.suptitle('temporal mean center variance', fontsize=20)
    plt.xlabel('variance', fontsize=18)
    plt.ylabel('patchlet count', fontsize=16)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images = wandb.Image(data, caption="Variance histogram")
    wandb.log({"variance histogram": images})

    # plot histogram
    patchlet_mean_variance = np.mean(np.array(patchlet_variance), 1)
    fig = plt.figure()
    plt.hist(patchlet_mean_variance, bins=30, range=[0, np.percentile(patchlet_mean_variance, 90)])
    fig.suptitle('temporal mean center mean variance', fontsize=20)
    plt.xlabel('mean variance', fontsize=18)
    plt.ylabel('patchlet count', fontsize=16)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images = wandb.Image(data, caption="Variance histogram")
    wandb.log({"Mean variance histogram": images})

    # plt.show()
    # patchlet_variance_scores = [[s] for s in patchlet_variance]
    # var_table = wandb.Table(data=patchlet_variance_scores, columns=["patchlet variance"])
    # wandb.log({'var_histogram': wandb.plot.histogram(var_table, "patchlet variance",
    #                                                  title="Patchlet variance distribution" )})


    columns = ["dataset name", "k", "sample_mode", "centroid_jitter", "degredation rate", "patchlet variance"]
    results_table = wandb.Table(columns=columns, data=[[dataset_name, k, sample_mode, add_centroid_jitter,
                                                        final_degradation_rate, final_average_patchlet_variance]])
    wandb.log({"Results summary": results_table})
    run.finish()

# Start sweep job.


if __name__ == "__main__":
    if sweep:
        wandb.agent(sweep_id, function=main, count=4)
    else:
        run = wandb.init()
        wandb.config.update(configuration)  # adds all of the arguments as config variables
        main()