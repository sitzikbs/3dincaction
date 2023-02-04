#!/usr/bin/env bash

GPU_IDX=0
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX

#DATASET_PATH='/data1/datasets/ANU_ikea_dataset_smaller/' # on remote
#DATASET_PATH='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/' # on local
DATASET_PATH='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller_clips/' # on local
INPUT_TYPE='pc'
CAMERA='dev3'
#PT_MODEL='charades' # when using images
DB_FILENAME='ikea_annotation_db_full'

#LOGDIR='/home/sitzikbs/Pycharm_projects/3dinaction/log/debug/'
LOGDIR='./log/pn1_debug/'
BATCH_SIZE=2
STEPS_PER_UPDATE=80
FRAMES_PER_CLIP=32
N_EPOCHS=2
N_GAUSSIANS=8         # for 3dmfv

PC_MODEL='pn1'
N_POINTS=4096
CACHE_CAPACITY=128
K=32

python train_i3d.py --dataset_path $DATASET_PATH --camera $CAMERA --batch_size $BATCH_SIZE --steps_per_update $STEPS_PER_UPDATE --logdir $LOGDIR --db_filename $DB_FILENAME --frames_per_clip $FRAMES_PER_CLIP --n_epochs $N_EPOCHS --input_type $INPUT_TYPE --n_points $N_POINTS --pc_model $PC_MODEL --n_gaussians $N_GAUSSIANS --cache_capacity $CACHE_CAPACITY --k $K
python test_i3d.py --dataset_path $DATASET_PATH --device $CAMERA --model_path $LOGDIR --batch_size 3 --db_filename $DATASET_PATH$DB_FILENAME --input_type $INPUT_TYPE --n_points $N_POINTS --pc_model $PC_MODEL --model '000025.pt' --n_gaussians $N_GAUSSIANS --sort_model $SORT_MODEL --k $K
python3 ../evaluation/evaluate_ikeaasm.py --results_path $LOGDIR'results/' --dataset_path $DATASET_PATH --mode vid
