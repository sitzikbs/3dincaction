from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import h5py
import visualization
import numpy as np
import random
import json

DATASET_N_POINTS=6890
class DfaustTActionDataset(Dataset):
    def __init__(self, dfaust_path, frames_per_clip=64, set='train', n_points=DATASET_N_POINTS, last_op='pad',
                 shuffle_points='once'):
        # self.sids = ['50002', '50004', '50007', '50009', '50020',
        # '50021', '50022', '50025', '50026', '50027']
        #TODO: add support for male set
        #TODO: add statistics computation (number of ids, number of actions etc)
        self.last_op = last_op
        if set == 'train':
            self.sids = ['50004', '50020', '50021', ]
        elif set =='test':
            self.sids = ['50022', '50025']
        else:
            raise ValueError("unsupported set")
        self.data_filename = os.path.join(dfaust_path, 'registrations_f.hdf5')
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
        self.frames_per_clip = frames_per_clip
        self.n_points = n_points
        self.shuffle_points = shuffle_points
        self.idxs = np.arange(DATASET_N_POINTS)
        if self.shuffle_points == 'once':
            random.shuffle(self.idxs)
        else:
            raise ValueError("Unknown shuffle protocol")


        self.vertices = []
        self.labels = []
        self.label_per_frame = []
        self.sid_per_seq = []
        self.n_frames_per_seq = {}
        self.faces = None
        self.clip_verts = None
        self.clip_labels = None
        self.seq_idx = None  # stores the sequence index for every clip
        self.subseq_pad = None  # stores the amount of padding for every clip


        self.load_data()
        self.clip_data()
        print(set+"set was loaded successfully and subdivided into clips.")


    def load_data(self):
        with h5py.File(self.data_filename, 'r') as f:
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

    def clip_data(self):

        for i, seq in enumerate(self.vertices):
            clip_vertices = list(self.chunks(seq,  self.frames_per_clip))
            frame_pad = np.zeros([len(clip_vertices)], dtype=np.int16)
            if clip_vertices[-1].shape[0] < self.frames_per_clip:
                if self.last_op == 'drop':
                    clip_vertices.pop()
                    frame_pad = frame_pad[:-2]
                else: #pad
                    frame_pad[-1] = int(self.frames_per_clip - len(clip_vertices[-1]))
                    clip_vertices[-1] = np.concatenate([clip_vertices[-2][len(clip_vertices[-1]):], clip_vertices[-1]])

            if self.clip_verts is None:
                self.clip_verts = clip_vertices
            else:
                self.clip_verts = np.concatenate([self.clip_verts, clip_vertices], axis=0)

            clip_labels = self.labels[i] * np.ones([len(clip_vertices), self.frames_per_clip])
            seq_idx = i * np.ones([len(clip_vertices)], dtype=np.int16)
            if self.clip_labels is None:
                self.clip_labels = clip_labels
                self.seq_idx = seq_idx
                self.subseq_pad = frame_pad
            else:
                self.clip_labels = np.concatenate([self.clip_labels, clip_labels], axis=0 )
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
            n_frames = self.n_frames_per_seq[seq]
            current_labels = np.zeros([n_frames, self.num_classes])
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

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.clip_verts)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):

        if self.shuffle_points =='shuffle_each':
            self.idxs = np.arange(DATASET_N_POINTS)
            random.shuffle(self.idxs)
        out_dict = {'points': self.clip_verts[idx][:, self.idxs[:self.n_points]], 'labels': self.clip_labels[idx],
                    'seq_idx': self.seq_idx[idx], 'padding': self.subseq_pad[idx]}
        return out_dict


if __name__ == '__main__':
    dataset = DfaustTActionDataset(dfaust_path='/home/sitzikbs/Datasets/dfaust/')
    test_loader = DataLoader(dataset, batch_size=4, num_workers=2, shuffle=True, drop_last=True)
    for batch, data in enumerate(test_loader):
        verts, labels = data['points'], data['labels']
        visualization.mesh_seq_vis(verts[0].detach().cpu().numpy(), dataset.faces,
                                   text=dataset.actions[int(labels.detach().cpu().numpy()[0][0])])