import torch
from torch.utils.data import DataLoader

from .IKEAEgoDatasetClips import IKEAEgoDatasetClips
from .DfaustDataset import DfaustActionClipsDataset
from .IKEAActionDatasetClips import IKEAActionDatasetClips

import i3d_utils as utils

def build_dataset(cfg, training=True):
    split = 'train' if training else 'test'
    cfg_data = cfg['DATA']
    if cfg_data.get('name') == 'DFAUST':
        data_augmentation = cfg['TRAINING'].get('aug') if split == 'train' else cfg['TESTING'].get('aug')
        dataset = DfaustActionClipsDataset(
            action_dataset_path=cfg_data['dataset_path'], frames_per_clip=cfg_data['frames_per_clip'], set=split,
            n_points=cfg_data['n_points'], shuffle_points=cfg_data['shuffle_points'], gender=cfg_data['gender'],
            data_augmentation=data_augmentation, noisy_data=cfg_data['noisy_data'],
        )
    elif cfg_data.get('name') == 'IKEA_ASM':
        dataset = IKEAActionDatasetClips(dataset_path=cfg_data['dataset_path'], set='test')
    elif cfg_data.get('name') == 'IKEA_EGO':
        dataset = IKEAEgoDatasetClips(dataset_path=cfg_data['dataset_path'], set=split)
    else:
        raise NotImplementedError
    return dataset


def build_dataloader(config, training=True, shuffle=False):
    dataset = build_dataset(config, training)

    num_workers = config['num_workers']
    batch_size = config['TRAINING'].get('batch_size') if training else config['TESTING'].get('batch_size')
    data_sampler = config['DATA'].get('data_sampler')

    split = 'train' if training else 'test'
    print("Number of clips in the {} set:{}".format(split, len(dataset)))

    if training and data_sampler == 'weighted':
        if config['DATA'].get('name') == 'DFAUST':
            weights = dataset.make_weights_for_balanced_classes()
        elif config['DATA'].get('name') == 'IKEA_ASM' or config['DATA'].get('name') == 'IKEA_EGO':
            weights = utils.make_weights_for_balanced_classes(dataset.clip_set, dataset.clip_label_count)
        else:
            raise NotImplementedError
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    else:
        sampler = None

    dataloader = DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        batch_size=batch_size,
        sampler=sampler,
    )
    return dataloader, dataset
