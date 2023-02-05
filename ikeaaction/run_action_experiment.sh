#!/usr/bin/env bash

GPU_IDX=0
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX

IDENTIFIER='debug'

#python train_i3d.py --identifier $IDENTIFIER
python test_i3d.py --identifier $IDENTIFIER --model_ckpt '000000.pt'
#python3 ../evaluation/evaluate_ikeaasm.py --results_path $LOGDIR'results/' --dataset_path $DATASET_PATH'/'$FRAMES_PER_CLIP --mode vid
