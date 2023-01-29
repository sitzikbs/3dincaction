import torch
import visualization
import numpy as np

# from DfaustDataset import DfaustActionClipsDataset as Dataset
from ikeaaction.IKEAActionDataset import IKEAActionVideoClipDataset as Dataset
from models.patchlets import PatchletsExtractor

# dataset_path = '/home/sitzikbs/Datasets/dfaust/'
dataset_path = '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/'
npoints = 128
k = 16
extract_pachlets = PatchletsExtractor(k=16, npoints=npoints)
# dataset = Dataset(dataset_path, frames_per_clip=64, set='train', n_points=1024,
#                         shuffle_points='fps_each_frame', gender='all', data_augmentation=[''] )
dataset = Dataset(dataset_path, frames_per_clip=64, set='train', n_points=1024, input_type='pc', camera='dev3',
                  mode='img', cache_capacity=1)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=8,
                                               pin_memory=True, shuffle=False, drop_last=True)

for batch_ind, data in enumerate(dataloader):
    # point_seq = data['points'].cuda()
    point_seq = data[0][..., :3, :].permute(0, 1, 3, 2).cuda()
    b, t, n, d = point_seq.shape
    patchlet_dict = extract_pachlets(point_seq)
    patch_idxs = patchlet_dict['patchlets']
    fps_idx = patchlet_dict['fps_idx']
    # set the color according to patchlets
    batch_ind = 2
    patch_mask = np.zeros([t, n, n])
    frame_ind = 0
    for frame_ind in range(t):
        for p_i in range(npoints):
            patch_mask[frame_ind, p_i][patch_idxs[batch_ind, 0, p_i].cpu().numpy()] = 1
            # patch_mask[frame_ind, 0][fps_idx[batch_ind, p_i].cpu().numpy()] = 1

    # visualization.pc_patchlet_points_vis(patchlet_dict['patchlet_points'][batch_ind].detach().cpu().numpy())
    visualization.pc_patchlet_vis(point_seq[batch_ind].cpu().numpy(), patch_mask.astype(np.float32))
    # visualization.pc_patchlet_vis(patchlet_dict['patchlet_points'][batch_ind, :, :, 0].cpu().numpy(), patch_mask.astype(np.float32))
    # visualization.pc_patchlet_patch_vis(patchlet_dict['patchlet_points'][batch_ind].cpu().numpy(), patchlet_dict['distances'][batch_ind].cpu().numpy())
#visualzie
# visualization.pc_seq_vis(verts[:, idxs[0:2048]])