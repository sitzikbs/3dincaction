#!/usr/bin/env bash

GPU_IDX=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX
export OMP_NUM_THREADS=8

export CUDA_PATH=/home/users/u1086420/anaconda3/envs/env_3DinAction_3_8/

IDENTIFIER=pn1_4d_basic_16bs_1steps_skip_connection_noisy_${GPU_IDX}_millbrae
CONFIG=./configs/config_dfaust_remote.yaml
LOGDIR=/data/sitzikbs/log/f64/baselines4paper/


python train_action_pred.py --identifier $IDENTIFIER --config $CONFIG --logdir $LOGDIR
python test_action_pred.py --identifier $IDENTIFIER --model_ckpt '000200.pt' --logdir $LOGDIR
python3 ../evaluation/evaluate.py --identifier $IDENTIFIER --logdir $LOGDIR
