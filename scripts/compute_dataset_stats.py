import matplotlib.pyplot as plt
from DfaustDataset import DfaustActionDataset, DfaustActionClipsDataset
import seaborn as sns
#TODO fix barplot visualization and export

# dataset = DfaustActionDataset(dfaust_path='/home/sitzikbs/datasets/dfaust/')
dataset = DfaustActionClipsDataset(action_dataset_path='/home/sitzikbs/datasets/dfaust/')
n_frames_per_label = dataset.get_dataset_statistics()
weights = dataset.make_weights_for_balanced_classes()
print(n_frames_per_label)
# sns.barplot({'actions': n_frames_per_label, '#frames': dataset.actions}, x='actions', y='#frames')
sns.barplot({'actions': n_frames_per_label, '#frames': dataset.action_dataset.actions}, x='actions', y='#frames')
plt.show()