#!/usr/bin/env bash

GPU_IDX=0
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX

IDENTIFIER='debug'
CONFIG='./configs/config_ikeaasm.yaml'
LOGDIR='./log/'

python train_i3d.py --identifier $IDENTIFIER --config $CONFIG --logdir $LOGDIR
#python test_i3d.py --identifier $IDENTIFIER --model_ckpt '000030.pt' --logdir $LOGDIR
#python3 ../evaluation/evaluate_ikeaasm.py --identifier $IDENTIFIER --logdir $LOGDIR
