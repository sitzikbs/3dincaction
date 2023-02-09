from DfaustDataset import DfaustActionClipsDataset as Dataset
import json

gender = 'male'
dataset_path = '/home/sitzikbs/Datasets/dfaust'
json_filename = '/home/sitzikbs/Datasets/dfaust/gt_segments_'+gender+'.json'
test_dataset = Dataset(dataset_path, frames_per_clip=64, set='test', n_points=2048, last_op='pad', gender=gender)
train_dataset = Dataset(dataset_path, frames_per_clip=64, set='train', n_points=2048, last_op='pad', gender=gender)

json_dict_to_write = {"version": "VERSION 1.3"}

results = {}
datasets = [test_dataset, train_dataset]
subset_dic = ['testing', 'training']
for j, dataset in enumerate(datasets):
    for i, subseq_name in enumerate(dataset.action_dataset.sid_per_seq):
        n_frames = int(len(dataset.action_dataset.vertices[i]))
        vid_name = subseq_name
        list_of_result_dicts = {"subset": subset_dic[j],
                                'annotation': [{"label": dataset.action_dataset.labels[i],  "segment": [0, n_frames]}]}
        results[vid_name] = list_of_result_dicts
json_dict_to_write["database"] = results

with open(json_filename, 'w') as outfile:
    json.dump(json_dict_to_write, outfile)