import os
import sys
sys.path.append('../ikeaego/')
from IKEAEgoDatasetClips import IKEAEgoDatasetClips as Dataset
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from itertools import compress

frames_per_clip = 32
dataset_path = '/home/sitzikbs/Datasets/ikeaego_small_clips_top10/' + str(frames_per_clip) + '/'
out_dir = './log/ego10/'
os.makedirs(out_dir, exist_ok=True)
out_dict = {'test':{}, 'train':{}}
with_names = False
data_mean_occ = np.array(0.) #Threshold, used 6000 for paper

for subset in out_dict.keys():
    dataset = Dataset(dataset_path, set=subset)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)
    print("printing info for {} set".format(subset))
    print("number of classes: {}".format(len(dataset.action_list)))
    print("Number of clips: {}".format(len(dataset)))
    print("Number of frames: {}".format(frames_per_clip *len(dataset)))
    print("Number of sequences: {}".format(len(dataset.video_list)))
    stats = dataset.get_dataset_statistics()
    actions = dataset.action_list
    out_dict[subset]['stats'] = stats
    out_dict[subset]['n_clips'] = len(dataset)
    out_dict[subset]['n_frames'] = frames_per_clip *len(dataset)




with open(os.path.join(out_dir, 'action_list.tex'), 'w') as f:
    for i, action in enumerate(actions):
        f.write(str(i) + '& ' + action + '// \n')

if with_names:
    action_labels = actions
    rot=90
    fname_add = '_labeled'
else:
    action_labels = np.arange(len(actions))
    rot=0
    fname_add = '_numbered'

action_labels = [x for _,x in sorted(zip(out_dict['train']['stats'],action_labels))]
out_dict['test']['stats'] = [x for _,x in sorted(zip(out_dict['train']['stats'],out_dict['test']['stats']))]
out_dict['train']['stats'].sort()

frequent_idxs = out_dict['train']['stats'] > data_mean_occ
regular_idxs = [not elem for elem in frequent_idxs]
#  plot frequent actions
bar_plot = pd.DataFrame({
  'train' : list(compress(out_dict['train']['stats'], frequent_idxs)),
  'test'  : list(compress(out_dict['test']['stats'], frequent_idxs))},
  index = list(compress(action_labels,  frequent_idxs)))

bar_plot.plot(kind='bar', stacked=True, color=['steelblue', 'orange'], width=0.8)
plt.xticks(rotation=rot)
csfont = {'fontname':'Times New Roman', 'size': 18}
plt.xlabel ('Action name',**csfont)
plt.ylabel ('# frames',**csfont)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(out_dir, 'ikeaego_frequent_stats'+fname_add +'.png'))
if not data_mean_occ == 0:
    # Plot all other actions

    # action_labels = [x for _,x in sorted(zip(out_dict['train']['stats'],action_labels))]
    # out_dict['test']['stats'] = [x for _,x in sorted(zip(out_dict['train']['stats'],out_dict['test']['stats']))]
    # out_dict['train']['stats'].sort()

    bar_plot = pd.DataFrame({
      'train' : list(compress(out_dict['train']['stats'], regular_idxs)),
      'test'  : list(compress(out_dict['test']['stats'], regular_idxs))},
      index = list(compress(action_labels,  regular_idxs)))

    bar_plot.plot(kind='bar', stacked=True, color=['steelblue', 'orange'], width=0.8)
    plt.xticks(rotation=rot)
    csfont = {'fontname':'Times New Roman', 'size': 18}
    plt.xlabel ('Action name',**csfont)
    plt.ylabel ('# frames',**csfont)


    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    N = len(actions)
    m = 0.2 # inch margin
    s = maxsize/plt.gcf().dpi*N+2*m
    margin = m/plt.gcf().get_size_inches()[0]
    plt.gcf().subplots_adjust(left=margin, right=1.-margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(out_dir, 'ikeaego_normal_stats' +fname_add +'.png'))