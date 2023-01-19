import torch
import visualization
import numpy as np

from DfaustDataset import DfaustActionClipsDataset as Dataset
from models.patchlets import PatchletsExtractor

dataset_path = '/home/sitzikbs/Datasets/dfaust/'
npoints = 128
extract_pachlets = PatchletsExtractor(npoints=npoints)
dataset = Dataset(dataset_path, frames_per_clip=64, set='train', n_points=1024,
                        shuffle_points='fps_each_frame', gender='all' )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=8,
                                               pin_memory=True, shuffle=False, drop_last=True)

for batch_ind, data in enumerate(dataloader):
    point_seq = data['points'].cuda()
    b, t, n, d = point_seq.shape
    patchlet_dict = extract_pachlets(point_seq)
    patch_idxs = patchlet_dict['patchlets']
    # set the color according to patchlets
    batch_ind = 0
    patch_mask = np.zeros([t, n, n])
    frame_ind = 0
    for frame_ind in range(t):
        for p_i in range(n):
            patch_mask[frame_ind, p_i][patch_idxs[batch_ind, frame_ind, p_i].cpu().numpy()] = 1

    visualization.pc_patchlet_vis(point_seq[batch_ind].cpu().numpy(), patch_mask.astype(np.float32))
    # visualization.pc_patchlet_vis(patchlet_dict['patchlet_points'][batch_ind, :, :, 0].cpu().numpy(), patch_mask.astype(np.float32))
    # visualization.pc_patchlet_patch_vis(patchlet_dict['patchlet_points'][batch_ind].cpu().numpy(), patchlet_dict['distances'][batch_ind].cpu().numpy())
#visualzie
# visualization.pc_seq_vis(verts[:, idxs[0:2048]])