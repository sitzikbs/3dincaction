GPU_IDX=0
NUM_THREADS=96
export OMP_NUM_THREADS=$NUM_THREADS
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export GOMP_CPU_AFFINITY="95-191"

#DATASET_PATH='/home/sitzikbs/Datasets/dfaust/'
DATASET_PATH='/data1/datasets/dfaust/'
MODEL='pn2_4d'
STEPS_PER_UPDATE=1
N_FRAMES=32
BATCH_SIZE=16
TEST_BATCH_SIZE=8

N_POINTS=1024
N_EPOCHS=200
POINTS_SHUFFLE='once'
SAMPLER='weighted'
AUGMENT=1
LOGDIR='./log/'$MODEL'_f'$N_FRAMES'_p'$N_POINTS'_shuffle_'$POINTS_SHUFFLE'_aug'$AUGMENT'_b'$BATCH_SIZE'_u'$STEPS_PER_UPDATE'/'
SET='test'


python3 train_action_pred.py --dataset_path $DATASET_PATH --pc_model $MODEL --steps_per_update $STEPS_PER_UPDATE --frames_per_clip $N_FRAMES --batch_size $BATCH_SIZE --shuffle_points $POINTS_SHUFFLE --logdir $LOGDIR --n_epochs $N_EPOCHS --n_points $N_POINTS --sampler $SAMPLER --data_augmentation $AUGMENT
python3 test_action_pred.py --dataset_path $DATASET_PATH --pc_model $MODEL --frames_per_clip $N_FRAMES --batch_size $TEST_BATCH_SIZE --shuffle_points $POINTS_SHUFFLE --n_points $N_POINTS --model '000200.pt' --model_path $LOGDIR --set $SET
python3 ./evaluation/evaluate.py --results_path $LOGDIR'results/' --dataset_path $DATASET_PATH --set $SET