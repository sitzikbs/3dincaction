from DfaustTActionDataset import DfaustTActionDataset as Dataset
import json

dataset_path = '/home/sitzikbs/datasets/dfaust'
json_filename = '/home/sitzikbs/datasets/dfaust/gt_segments.json'
test_dataset = Dataset(dataset_path, frames_per_clip=64, set='test', n_points=2048, last_op='pad')
train_dataset = Dataset(dataset_path, frames_per_clip=64, set='train', n_points=2048, last_op='pad')

json_dict_to_write = {"version": "VERSION 1.3"}

results = {}
datasets = [test_dataset, train_dataset]
subset_dic = ['testing', 'training']
for j, dataset in enumerate(datasets):
    for i, subseq_name in enumerate(dataset.sid_per_seq):
        n_frames = int(len(dataset.vertices[i]))
        vid_name = subseq_name
        list_of_result_dicts = {"subset": subset_dic[j],
                                'annotation': [{"label": dataset.labels[i],  "segment": [0, n_frames]}]}
        results[vid_name] = list_of_result_dicts
json_dict_to_write["database"] = results

with open(json_filename, 'w') as outfile:
    json.dump(json_dict_to_write, outfile)