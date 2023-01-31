import torch
import visualization
import numpy as np
from DfaustDataset import DfaustActionClipsDataset
from ikeaaction.IKEAActionDataset import IKEAActionVideoClipDataset
#

from models.patchlets import PatchletsExtractor

dataset_name = 'ikea'
npoints = 128
k = 16
if dataset_name == 'ikea':
    dataset_path = '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/'
    dataset = IKEAActionVideoClipDataset(dataset_path, frames_per_clip=64, set='train', n_points=1024, input_type='pc', camera='dev3',
                      mode='img', cache_capacity=1)
else:
    dataset_path = '/home/sitzikbs/Datasets/dfaust/'
    dataset = DfaustActionClipsDataset(dataset_path, frames_per_clip=64, set='train', n_points=1024,
                            shuffle_points='fps_each_frame', gender='all', data_augmentation=[''] )


extract_pachlets = PatchletsExtractor(k=16, npoints=npoints)


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


    visualization.pc_patchlet_points_vis(patchlet_dict['patchlet_points'][batch_ind].detach().cpu().numpy())
    # visualization.pc_patchlet_vis(point_seq[batch_ind].cpu().numpy(), patchlet_dict['patchlet_points'][batch_ind].cpu().numpy())
    # visualization.pc_patchlet_vis(patchlet_dict['patchlet_points'][batch_ind, :, :, 0].cpu().numpy(), patch_mask.astype(np.float32))
    # visualization.pc_patchlet_patch_vis(patchlet_dict['patchlet_points'][batch_ind].cpu().numpy(), patchlet_dict['distances'][batch_ind].cpu().numpy())
