from torch.utils.data import Dataset
import pickle
import os


class IKEAActionDatasetClips(Dataset):
    """
    IKEA Action Dataset class with pre-saved clips into pickles
    """

    def __init__(self, dataset_path, set):
        #TODO add support for point cloud downsampling using FPS and random sampling
        self.dataset_path = dataset_path
        self.set = set
        self.files_path = os.path.join(dataset_path, set)
        self.file_list = self.absolute_file_paths(self.files_path)
        self.file_list.sort()
        # backwards compatibility
        with open(os.path.join(self.dataset_path, set+'_aux.pickle'), 'rb') as f:
            aux_data = pickle.load(f)
        self.clip_set = aux_data['clip_set']
        self.clip_label_count = aux_data['clip_label_count']
        self.num_classes = aux_data['num_classes']
        self.video_list = aux_data['video_list']
        self.action_list = aux_data['action_list']
        self.frames_per_clip = aux_data['frames_per_clip']
        self.action_labels = aux_data['action_labels']
        print("{}set contains {} clips".format(set, len(self.file_list)))

    def absolute_file_paths(self, directory):
        path = os.path.abspath(directory)
        return [entry.path for entry in os.scandir(path) if entry.is_file()]

    def get_num_seq(self):
        return len(self.video_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        with open(self.file_list[index], 'rb') as f:
            data = pickle.load(f)
            # normalize RGB values to 0-1
            inputs = data.get('inputs')
            inputs[:, 3:, :] /= 255.0
            data['inputs'] = inputs
        return data
