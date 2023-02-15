#!/usr/bin/env bash

GPU_IDX=0
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX
export OMP_NUM_THREADS=8

IDENTIFIER='pn1'
CONFIG='./configs/config_ikeaego_remote.yaml'
LOGDIR='./log/f32_baseline4paper/'

python train_i3d.py --identifier $IDENTIFIER --config $CONFIG --logdir $LOGDIR
python test_i3d.py --identifier $IDENTIFIER --model_ckpt '000030.pt' --logdir $LOGDIR
python3 ../evaluation/evaluate_ikeaasm.py --identifier $IDENTIFIER --logdir $LOGDIR