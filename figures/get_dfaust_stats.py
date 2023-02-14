import os
import sys
sys.path.append('../dfaust/')
from DfaustDataset import DfaustActionClipsDataset as Dataset
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset_path = '/home/sitzikbs/Datasets/dfaust/'
frames_per_clip = 64
gender = 'all'
out_dir = './log'
out_dict = {'train':{}, 'test':{}}
for subset in out_dict.keys():
    dataset = Dataset(dataset_path, frames_per_clip=frames_per_clip, set=subset, n_points=1024, last_op='pad',
                      shuffle_points='each', gender=gender)
    test_loader = DataLoader(dataset, batch_size=2, num_workers=1, shuffle=False, drop_last=True)

    print("Number of clips: {}".format(len(dataset)))
    print("Number of frames: {}".format(frames_per_clip *len(dataset)))
    print("Number of sequences: {}".format(len(dataset.action_dataset)))
    stats = dataset.get_dataset_statistics()
    actions = dataset.action_dataset.actions
    out_dict[subset]['stats'] = stats
    out_dict[subset]['n_clips'] = len(dataset)
    out_dict[subset]['n_frames'] = frames_per_clip *len(dataset)


bar_plot = pd.DataFrame({
  'train' : out_dict['train']['stats'],
  'test'  : out_dict['test']['stats']},
  index = actions)

bar_plot.plot(kind='bar', stacked=True, color=['steelblue', 'orange'], width=0.8)
plt.xticks(rotation=45, ha='right')
csfont = {'fontname':'Times New Roman', 'size': 18}
plt.xlabel ('Action name',**csfont)
plt.ylabel ('# frames',**csfont)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(out_dir, 'dfaust_stats.png'))