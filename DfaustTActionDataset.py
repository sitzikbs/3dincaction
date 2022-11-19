from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import h5py
import visualization
import numpy as np

class DfaustTActionDataset(Dataset):
    def __init__(self, dfaust_path, frames_per_clip=64, set='train', n_points=2048):
        # self.sids = ['50002', '50004', '50007', '50009', '50020',
        # '50021', '50022', '50025', '50026', '50027']
        #TODO: add support for male set
        #TODO: add support for selecting number of points
        #TODO: add statistics computation (number of ids, number of actions etc)
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

        self.vertices = []
        self.labels = []
        self.faces = None
        self.clip_verts = None
        self.clip_labels = None

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

    def clip_data(self):

        for i, seq in enumerate(self.vertices):
            clip_vertices = list(self.chunks(seq,  self.frames_per_clip))
            if clip_vertices[-1].shape[0] < self.frames_per_clip:
                clip_vertices.pop()
            if self.clip_verts is None:
                self.clip_verts = clip_vertices
            else:
                self.clip_verts = np.concatenate([self.clip_verts, clip_vertices], axis=0)

            clip_labels = self.labels[i] * np.ones([len(clip_vertices), self.frames_per_clip])
            if self.clip_labels is None:
                self.clip_labels = clip_labels
            else:
                self.clip_labels = np.concatenate([self.clip_labels, clip_labels], axis=0 )


    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.clip_verts)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        out_dict = {'points': self.clip_verts[idx], 'labels': self.clip_labels[idx]}
        return out_dict


if __name__ == '__main__':
    dataset = DfaustTActionDataset(dfaust_path='/home/sitzikbs/Datasets/dfaust/')
    test_loader = DataLoader(dataset, batch_size=4, num_workers=2, shuffle=True, drop_last=True)
    for batch, data in enumerate(test_loader):
        verts, labels = data['points'], data['labels']
        visualization.mesh_seq_vis(verts[0].detach().cpu().numpy(), dataset.faces,
                                   text=dataset.actions[int(labels.detach().cpu().numpy()[0][0])])