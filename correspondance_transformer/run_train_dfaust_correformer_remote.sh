GPU_IDX=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX


NPOINTS=1024
BATCH_SIZE=40
NHEADS=8
DIM=1024
DFF=1024
DATASET_PATH='/data1/datasets/dfaust/'
EXP_ID='jitter0.01NoSampler'
AUG='jitter'

taskset -c 128-160 python3 train_dfaust_correformer.py --dataset_path $DATASET_PATH --dim $DIM --n_heads $NHEADS --batch_size $BATCH_SIZE --n_points $NPOINTS --d_feedforward $DFF --exp_id $EXP_ID --aug $AUG