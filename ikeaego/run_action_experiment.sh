#!/usr/bin/env bash

GPU_IDX=0
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX

IDENTIFIER='pn1_8bs_20steps_cuda1_millbrae'
CONFIG='./configs/config_ikeaego.yaml'
LOGDIR='./log/'

#python train_ego.py --identifier $IDENTIFIER --config $CONFIG --logdir $LOGDIR
#python test_ego.py --identifier $IDENTIFIER --model_ckpt '000000.pt' --logdir $LOGDIR
python3 ../evaluation/evaluate_ikeaego.py --identifier $IDENTIFIER --logdir $LOGDIR
