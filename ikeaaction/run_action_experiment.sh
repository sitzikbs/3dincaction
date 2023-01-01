#!/usr/bin/env bash

GPU_IDX=0
NUM_THREADS=96
export OMP_NUM_THREADS=$NUM_THREADS
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export GOMP_CPU_AFFINITY="95-191"
#DATASET_PATH='/data1/datasets/ANU_ikea_dataset_smaller/' # on remote
DATASET_PATH='/home/sitzikbs/datasets/ANU_ikea_dataset_smaller/' # on local
INPUT_TYPE='pc'
CAMERA='dev3'
#PT_MODEL='charades' # when using images
DB_FILENAME='ikea_annotation_db_full'

#LOGDIR='/home/sitzikbs/Pycharm_projects/3dinaction/log/debug/'
LOGDIR='./log/pn1_baseline/'
BATCH_SIZE=10
STEPS_PER_UPDATE=16
FRAMES_PER_CLIP=32
N_EPOCHS=31
N_POINTS=4096
PC_MODEL='pn1'
USE_POINTLETTES=0     # deprecated
POINTLET_MODE='none'  # deprecated
N_GAUSSIANS=8         # for 3dmfv
CORREFORMER='none' # path or 'none'
python train_i3d.py --dataset_path $DATASET_PATH --camera $CAMERA --batch_size $BATCH_SIZE --steps_per_update $STEPS_PER_UPDATE --logdir $LOGDIR --db_filename $DB_FILENAME --frames_per_clip $FRAMES_PER_CLIP --n_epochs $N_EPOCHS --input_type $INPUT_TYPE --n_points $N_POINTS --pc_model $PC_MODEL --use_pointlettes $USE_POINTLETTES --pointlet_mode $POINTLET_MODE --n_gaussians $N_GAUSSIANS --correformer $CORREFORMER
python test_i3d.py --dataset_path $DATASET_PATH --device $CAMERA --model_path $LOGDIR --batch_size 3 --db_filename $DATASET_PATH$DB_FILENAME --input_type $INPUT_TYPE --n_points $N_POINTS --pc_model $PC_MODEL --use_pointlettes $USE_POINTLETTES --pointlet_mode $POINTLET_MODE --model '000025.pt' --n_gaussians $N_GAUSSIANS --correformer $CORREFORMER
python3 ./evaluation/evaluate.py --results_path $LOGDIR'results/' --dataset_path $DATASET_PATH --mode vid