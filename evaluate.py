# Author: Yizhak Ben-Shabat (Itzik), 2020
# evaluate action recognition performance using acc and mAp

import os
import argparse
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import utils
from evaluation import eval_utils
from datasets import build_dataset
import matplotlib.pyplot as plt
from evaluation.eval_detection import ANETdetection
from evaluation.eval_classification import ANETclassification
import sklearn
import yaml
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='../log/', help='path to model save dir')
parser.add_argument('--identifier', type=str, default='debug', help='unique run identifier')
args = parser.parse_args()

cfg = yaml.safe_load(open(os.path.join(args.logdir, args.identifier, 'config.yaml')))
run = wandb.init(entity=cfg['WANDB']['entity'], project=cfg['WANDB']['project'], id=cfg['WANDB']['id'], resume='must')
wandb.define_metric("eval/step")
wandb.define_metric("eval/*", step_metric="eval/step")

results_path = os.path.join(args.logdir, args.identifier, 'results/')
subset = cfg['TESTING']['set']
data_name = cfg['DATA']['name']

# load the gt and predicted data
training_ = True if subset == 'train' else False
dataset = build_dataset(cfg, training=training_)
gt_labels = dataset.action_labels

if data_name == 'DFAUST':
    gender = cfg['DATA']['gender']
    gt_json_path = os.path.join(cfg['DATA']['dataset_path'], 'gt_segments_'+gender+'.json')
elif data_name == 'IKEA_EGO' or data_name == 'IKEA_ASM':
    gt_json_path = os.path.join(cfg['DATA']['dataset_path'], 'gt_segments.json')
else:
    raise NotImplementedError

results_json = os.path.join(results_path, subset + '_action_segments.json')
results_npy = os.path.join(results_path, subset + '_pred.npy')

# load the predicted data
pred_data = np.load(results_npy, allow_pickle=True).item()
pred_labels = pred_data['pred_labels']
logits = pred_data['logits']

alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6,
                  0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
# compute action localization mAP
anet_detection = ANETdetection(gt_json_path, results_json,
                               subset='testing', tiou_thresholds=alpha,
                               verbose=True, check_status=True)
anet_detection.evaluate()

localization_score_str = "Action localization scores: \n" \
                         "Average mAP= {} \n".format(anet_detection.average_mAP) + \
                         "alpha = " + " & ".join(str(alpha).split()) + "\n " \
                         "mAP scores =" +\
                         " & ".join(str(np.around(anet_detection.mAP, 2)).split()) + "\n "

# Compute classification mAP
anet_classification = ANETclassification(gt_json_path, results_json, subset='testing', verbose=True)
anet_classification.evaluate()
mAP = anet_classification.ap.mean()
classification_score_str = "Action classification scores: \n" \
                            "mAP = {} \n".format(mAP)

# Compute accuracy
acc1_per_vid = []
acc3_per_vid = []

gt_single_labels = []
for vid_idx in range(len(logits)):
    effective_frames = len(logits[vid_idx])  # avoid padding for ego since last frames are not necessary
    single_label_per_frame = torch.argmax(torch.tensor(gt_labels[vid_idx][:effective_frames]), dim=1)  # avoid padding for ego since last frames are not necessary

    acc1, acc3 = eval_utils.accuracy(torch.tensor(logits[vid_idx]), single_label_per_frame, topk=(1, 3))
    acc1_per_vid.append(acc1.item())
    acc3_per_vid.append(acc3.item())
    gt_single_labels.append(single_label_per_frame)
top1, top3 = round(np.mean(acc1_per_vid), 2), round(np.mean(acc3_per_vid), 2)
scores_str = 'top1, top3 accuracy: {} & {}\n'.format(top1, top3)

print(scores_str)
balanced_acc_per_vid = []

for vid_idx in range(len(logits)):
    effective_frames = len(logits[vid_idx])  # avoid padding for ego since last frames are not necessary
    single_label_per_frame = torch.argmax(torch.tensor(gt_labels[vid_idx][:effective_frames]), dim=1)
    acc = sklearn.metrics.balanced_accuracy_score(single_label_per_frame, np.argmax(logits[vid_idx], 1),
                                                  sample_weight=None, adjusted=False)
    balanced_acc_per_vid.append(acc)

balanced_score = round(np.mean(balanced_acc_per_vid)*100, 2)
balanced_score_str = 'balanced accuracy: {}'.format(balanced_score)
print(balanced_score_str)


# output the dataset total score
with open(os.path.join(results_path, 'scores.txt'), 'w') as file:
    file.writelines(localization_score_str)
    file.writelines(classification_score_str)
    file.writelines(scores_str)
    file.writelines(balanced_score_str)

# Compute confusion matrix
print('Comptuing confusion matrix...')

c_matrix = confusion_matrix(np.concatenate(gt_single_labels).ravel(), np.concatenate(pred_labels).ravel(),
                            labels=range(dataset.num_classes))
class_names = utils.squeeze_class_names(dataset.action_list)

fig, ax = utils.plot_confusion_matrix(
    cm=c_matrix,
    target_names=class_names,
    title='Confusion matrix',
    cmap=None,
    normalize=True,
)
img_out_filename = os.path.join(results_path, 'confusion_matrix.png')
plt.savefig(img_out_filename)
img = plt.imread(img_out_filename)

columns = ["top 1", "top 3", "macro", "mAP"]
results_table = wandb.Table(columns=columns, data=[[top1, top3, balanced_score, mAP]])
images = wandb.Image(img, caption="Confusion matrix")
wandb.log({"eval/confusion matrix": images, "eval/Results summary": results_table})

