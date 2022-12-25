from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import h5py
import visualization
import numpy as np
import random
import json
import pc_transforms as transforms
import models.correformer as cf
# from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import torch
from tqdm import tqdm

DATASET_N_POINTS = 6890

class DfaustActionClipsDataset(Dataset):
    def __init__(self, action_dataset_path, frames_per_clip=64, set='train', n_points=DATASET_N_POINTS, last_op='pad',
                 shuffle_points='once', data_augmentation=[], aug_params_dict={'sigma': 0.005},
                 gender='female', nn_sample_ratio=1):
        self.action_dataset = DfaustActionDataset(action_dataset_path, set, gender=gender)
        self.frames_per_clip = frames_per_clip
        self.n_points = n_points
        self.shuffle_points = shuffle_points
        self.last_op = last_op
        self.data_augmentation = data_augmentation
        self.aug_params_dict = aug_params_dict
        self.nn_sample_ratio = nn_sample_ratio # subsample the points

        self.clip_verts = None
        self.clip_labels = None
        self.subseq_pad = None  # stores the amount of padding for every clip
        self.idxs = np.arange(DATASET_N_POINTS)
        self.randomizer = random.Random(0)
        if self.shuffle_points == 'once':
            self.randomizer.shuffle(self.idxs)
        elif self.shuffle_points == 'none' or self.shuffle_points == 'each' or self.shuffle_points == 'each_frame':
            pass
        else:
            raise ValueError("Unknown shuffle protocol")

        self.clip_data()


    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def clip_data(self):

        for i, seq in enumerate(self.action_dataset.vertices):
            clip_vertices = list(self.chunks(seq, self.frames_per_clip))
            frame_pad = np.zeros([len(clip_vertices)], dtype=np.int16)
            if clip_vertices[-1].shape[0] < self.frames_per_clip:
                if self.last_op == 'drop':
                    clip_vertices.pop()
                    frame_pad = frame_pad[:-2]
                elif self.last_op == 'pad':  # pad
                    frame_pad[-1] = int(self.frames_per_clip - len(clip_vertices[-1]))
                    clip_vertices[-1] = np.concatenate(
                        [clip_vertices[-2][len(clip_vertices[-1]):], clip_vertices[-1]])

            if self.clip_verts is None:
                self.clip_verts = clip_vertices
            else:
                self.clip_verts = np.concatenate([self.clip_verts, clip_vertices], axis=0)

            clip_labels = self.action_dataset.labels[i] * np.ones([len(clip_vertices), self.frames_per_clip])
            seq_idx = i * np.ones([len(clip_vertices)], dtype=np.int16)
            if self.clip_labels is None:
                self.clip_labels = clip_labels
                self.seq_idx = seq_idx
                self.subseq_pad = frame_pad
            else:
                self.clip_labels = np.concatenate([self.clip_labels, clip_labels], axis=0)
                self.seq_idx = np.concatenate([self.seq_idx, seq_idx], axis=0)
                self.subseq_pad = np.concatenate([self.subseq_pad, frame_pad], axis=0)

    def get_actions_labels_from_json(self, json_filename, mode='gt'):
        """

         Loads a label segment .json file (ActivityNet format
          http://activity-net.org/challenges/2020/tasks/anet_localization.html) and converts to frame labels for evaluation

        Parameters
        ----------
        json_filename : output .json file name (full path)
        device: camera view to use
        Returns
        -------
        frame_labels: one_hot frame labels (allows multi-label)
        """
        labels = []
        with open(json_filename, 'r') as json_file:
            json_dict = json.load(json_file)

        if mode == 'gt':
            video_results = json_dict["database"]
        else:
            video_results = json_dict["results"]

        for seq in video_results:
            n_frames = self.action_dataset.n_frames_per_seq[seq]
            current_labels = np.zeros([n_frames, self.action_dataset.num_classes])
            if mode == 'gt':
                segments = video_results[seq]['annotation']
            else:
                segments = video_results[seq]
            for segment in segments:
                action_idx = segment["label"]
                start = segment['segment'][0]
                end = segment['segment'][1]
                current_labels[start:end, action_idx] = 1
            labels.append(current_labels)
        return labels

    def get_dataset_statistics(self):
        n_frames_per_label = np.zeros(len(self.action_dataset.actions))
        for i, clip in enumerate(self.clip_labels):
            for j, frame_label in enumerate(clip):
                n_frames_per_label[int(frame_label)] = n_frames_per_label[int(frame_label)] + 1
        return n_frames_per_label

    def make_weights_for_balanced_classes(self):
        """ compute the weight per clip for the weighted random sampler"""
        n_clips = len(self.clip_verts)
        nclasses = len(self.action_dataset.actions)
        n_frames_per_label = self.get_dataset_statistics()
        N = n_frames_per_label.sum()
        weight_per_class = [0.] * nclasses
        for i in range(nclasses):
            weight_per_class[i] = N / float(n_frames_per_label[i])

        weight = [0] * n_clips
        for idx, clip in enumerate(self.clip_labels):
            clip_one_hot = np.zeros((clip.size, nclasses))
            clip_one_hot[np.arange(clip.size), clip.astype(int)] = 1
            clip_label_sum = clip_one_hot.sum(axis=0)

            if clip_label_sum.sum() == 0:
                print("Unlabeled clip!!!")
            ratios = clip_label_sum / clip_label_sum.sum()
            weight[idx] = np.dot(weight_per_class, ratios)
        return weight

    def augment_points(self, points):
        if self.data_augmentation:
            out_points = points
            if 'scale' in self.data_augmentation:
                out_points = transforms.random_scale_point_cloud(out_points, scale_low=0.8, scale_high=1.25)
            if 'rotate' in self.data_augmentation:
                out_points = transforms.rotate_perturbation_point_cloud(out_points, angle_sigma=0.06, angle_clip=0.18)
            # jitter is handled externally
            # if 'jitter' in self.data_augmentation:
            #     out_points = transforms.jitter_point_cloud(out_points, sigma=self.aug_params_dict['sigma'],
            #                                                clip=5 * self.aug_params_dict['sigma'])
            if 'translate' in self.data_augmentation:
                out_points = transforms.shift_point_cloud(out_points, shift_range=0.1)
        else:
            out_points = points
        return out_points

    def nn_sampler(self, points):
        # sample more points that are not well corresponded using nn
        if self.nn_sample_ratio < 1:
            n_final_points = int(self.nn_sample_ratio * DATASET_N_POINTS)
            # reorder the points to put the informative points first, then clip off points that nn gets well
            tree = cKDTree(points[0])
            _, max_ind = tree.query(points[1])
            nn_idx = max_ind == np.arange(DATASET_N_POINTS)
            good_points = points[:, nn_idx]
            bad_points = points[:, np.logical_not(nn_idx)]
            out_points = np.concatenate([bad_points, good_points], axis=1)
            out_points = out_points[:, :n_final_points, :]
        else:
            out_points = points
        return out_points

    def __len__(self):
        return len(self.clip_verts)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        points_seq = self.nn_sampler(self.clip_verts[idx])
        # points_seq = self.clip_verts[idx]
        n_initial_points = int(DATASET_N_POINTS * self.nn_sample_ratio)
        if self.shuffle_points == 'each':
            shuffled_idxs = np.arange(n_initial_points)
            random.shuffle(shuffled_idxs)
            shuffled_idxs = shuffled_idxs[:self.n_points]
            points_seq = points_seq[:, shuffled_idxs]
        elif self.shuffle_points == 'each_frame':
            shuffled_idxs = np.arange(n_initial_points)
            random.shuffle(shuffled_idxs)
            points_seq = points_seq[:, shuffled_idxs[:self.n_points]]
            shuffled_idxs = np.array([np.random.permutation(np.arange(self.n_points)) for _ in range(self.frames_per_clip-1)])[:, :, None]
            shuffled_idxs = np.insert(shuffled_idxs, 0, np.arange(self.n_points)[None, :, None], axis=0) # make sure thefirst frame indices are unchanged (they are refs)
            points_seq = np.take_along_axis(points_seq, shuffled_idxs[:, :self.n_points], axis=1)
        else:
            shuffled_idxs = np.arange(n_initial_points)[:self.n_points] # not shuffled
            points_seq = points_seq[shuffled_idxs[:self.n_points], :]


        out_points = self.augment_points(points_seq)


        out_dict = {'points': out_points, 'labels': self.clip_labels[idx],
                    'seq_idx': self.seq_idx[idx], 'padding': self.subseq_pad[idx],
                    'corr_gt': shuffled_idxs, 'idx': idx}
        return out_dict

class DfaustActionDataset(Dataset):
    # A dataset class for action sequences. No batch support since sequences are not of equal lengths.
    # batch is supported in the clip based version that clips the sequences into equal length clips
    def __init__(self, dfaust_path,  set='train', gender='female', n_points=DATASET_N_POINTS, shuffle_points='none'):

        dataset_subdivision_dict = {'train': {'female': {'sids': ['50004', '50020', '50021'],
                                                         'filnames': [os.path.join(dfaust_path, 'registrations_f.hdf5')]},
                                              'male': {'sids': ['50002', '50007', '50009'],
                                                         'filnames': [os.path.join(dfaust_path, 'registrations_m.hdf5')]},
                                              'all': {'sids': ['50004', '50020', '50021', '50002', '50007', '50009'],
                                                         'filnames': [os.path.join(dfaust_path, 'registrations_f.hdf5'),
                                                                      os.path.join(dfaust_path, 'registrations_m.hdf5')]}
                                              },
                                    'test': {'female': {'sids': ['50022', '50025'],
                                                        'filnames': [os.path.join(dfaust_path, 'registrations_f.hdf5')]},
                                             'male': {'sids': ['50026', '50027'],
                                                      'filnames': [os.path.join(dfaust_path, 'registrations_m.hdf5')]},
                                             'all': {'sids': ['50022', '50025', '50026', '50027'],
                                                     'filnames': [os.path.join(dfaust_path, 'registrations_f.hdf5'),
                                                                  os.path.join(dfaust_path, 'registrations_m.hdf5')]}
                                             }
                                    }

        self.sids = dataset_subdivision_dict[set][gender]['sids']
        self.data_filename =  dataset_subdivision_dict[set][gender]['filnames']
        self.actions = ['chicken_wings',
                    'hips',
                    'jiggle_on_toes',
                    'jumping_jacks',
                    'knees',
                    'light_hopping_loose',
                    'light_hopping_stiff',
                    'one_leg_jump',
                    'one_leg_loose',
                    'punching',
                    'running_on_spot',
                    'shake_arms',
                    'shake_hips',
                    'shake_shoulders']
        self.num_classes = len(self.actions)
        self.vertices = []
        self.labels = []
        self.label_per_frame = []
        self.sid_per_seq = []
        self.n_frames_per_seq = {}
        self.faces = None
        self.seq_idx = None  # stores the sequence index for every clip

        self.n_points = n_points
        self.shuffle_points = shuffle_points
        self.idxs = np.arange(DATASET_N_POINTS)
        self.randomizer = random.Random(0)
        if self.shuffle_points == 'once':
            self.randomizer.shuffle(self.idxs)
        elif self.shuffle_points == 'none' or self.shuffle_points == 'each' or self.shuffle_points == 'each_frame':
            pass
        else:
            raise ValueError("Unknown shuffle protocol")

        self.load_data()

        print(set+"set was loaded successfully and subdivided into clips.")


    def load_data(self):
        for filename in self.data_filename:
            with h5py.File(filename, 'r') as f:
                for id in self.sids:
                    for i, action in enumerate(self.actions):
                        sidseq = id + '_' + action
                        if sidseq in f:
                            vertices = f[sidseq][()].transpose([2, 0, 1])
                            faces = f['faces'][()]
                            #scale to unit sphere centered at the first frame
                            t = np.mean(vertices[0], axis=0)
                            s = np.linalg.norm(np.max(np.abs(vertices[0] - t), axis=0))
                            vertices = (vertices - t )/ s

                            self.vertices.append(vertices)
                            if self.faces is None:
                                self.faces = faces
                            self.labels.append(i)
                            self.label_per_frame.append(i*np.ones(len(vertices)))
                            self.sid_per_seq.append(sidseq)
                            self.n_frames_per_seq[sidseq] = len(vertices)

    def get_dataset_statistics(self):
        n_frames_per_label = np.zeros(len(self.actions))
        for i, seq in enumerate(self.label_per_frame):
            for j, frame_label in enumerate(seq):
                n_frames_per_label[int(frame_label)] = n_frames_per_label[int(frame_label)] + 1
        return n_frames_per_label


    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.vertices)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):

        if self.shuffle_points == 'each':
            shuffled_idxs = np.arange(DATASET_N_POINTS)
            random.shuffle(shuffled_idxs)
            shuffled_idxs = shuffled_idxs[:self.n_points]
            points_seq = self.vertices[idx][:, shuffled_idxs]
            shuffled_idxs = np.arange(self.n_points)  # the new indices match throghout the sequence
        elif self.shuffle_points == 'each_frame':
            shuffled_idxs = np.arange(DATASET_N_POINTS)
            random.shuffle(shuffled_idxs)
            points_seq = self.vertices[idx][:, shuffled_idxs[:self.n_points]]
            shuffled_idxs = np.array([np.random.permutation(np.arange(self.n_points)) for _ in range(len(self.vertices[idx])-1)])[:, :, None]
            shuffled_idxs = np.insert(shuffled_idxs, 0, np.arange(self.n_points)[None, :, None], axis=0) # make sure thefirst frame indices are unchanged (they are refs)
            points_seq = np.take_along_axis(points_seq, shuffled_idxs[:, :self.n_points], axis=1)
        else:
            # this will give a partial point cloud of the human since the order of the points covers different regions
            shuffled_idxs = np.arange(DATASET_N_POINTS)[:self.n_points]  # not shuffled
            points_seq = self.vertices[idx][:, shuffled_idxs, :]

        out_dict = {'points': points_seq, 'labels': self.labels[idx], 'seq_idx': idx, 'corr_gt': shuffled_idxs}
        return out_dict


if __name__ == '__main__':
    # dataset = DfaustActionDataset(dfaust_path='/home/sitzikbs/Datasets/dfaust/')
    correformer_path = './transformer_toy_example/log/dfaust_N1024_d1024h16_lr1e-05bs16_/000000.pt'
    dataset = DfaustActionClipsDataset(action_dataset_path='/home/sitzikbs/Datasets/dfaust/',
                                       frames_per_clip=16, set='train', n_points=1024, last_op='pad',
                                       shuffle_points='each_frame')
    test_loader = DataLoader(dataset, batch_size=2, num_workers=1, shuffle=False, drop_last=True)#,
                             # multiprocessing_context='spawn')


    correformer = cf.get_correformer(correformer_path)

    for batch, data in enumerate(test_loader):
        verts, labels = data['points'], data['labels']
        # visualization.mesh_seq_vis(verts[0].detach().cpu().numpy(), dataset.faces,
        #                            text=dataset.actions[int(labels.detach().cpu().numpy()[0])])
        corr_gt = data['corr_gt']

        sorted_verts, corr_pred = cf.sort_points(correformer, verts)

        diff = torch.abs(corr_gt[[0]].squeeze() - corr_pred[[0]].detach().cpu().numpy())
        points_color = np.repeat(np.arange(len(verts[0][0]))[None, :], len(verts[0]), axis=0)
        points_color = np.repeat(np.arange(len(verts[0][0]))[None, :], len(verts[0]), axis=0)
        visualization.pc_seq_vis(verts[0].detach().cpu().numpy(),
                                text=np.take(dataset.action_dataset.actions,
                                                labels.detach().cpu().numpy()[0].astype(int)),
                                 color=points_color)