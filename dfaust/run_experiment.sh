#!/usr/bin/env bash

GPU_IDX=0
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX

IDENTIFIER='debug'
CONFIG='./configs/config_dfaust.yaml'
#LOGDIR='./log/jitter/f32/baselines/'
LOGDIR='./log/'

python train_action_pred.py --identifier $IDENTIFIER --config $CONFIG --logdir $LOGDIR
python test_action_pred.py --identifier $IDENTIFIER --model_ckpt '000200.pt' --logdir $LOGDIR
python3 ./evaluation/evaluate.py --identifier $IDENTIFIER --logdir $LOGDIR