GPU_IDX=0
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX


#DATASET_PATH='/home/sitzikbs/Datasets/dfaust/'
DATASET_PATH='/data1/datasets/dfaust/'
MODEL='pn1_4d'
STEPS_PER_UPDATE=1
N_FRAMES=64
BATCH_SIZE=16
TEST_BATCH_SIZE=4
TEST_ITER=200
GENDER='all'

N_POINTS=1024
N_EPOCHS=200
POINTS_SHUFFLE='fps_each'
SAMPLER='weighted'
AUGMENT=1


SET='test'
GT_JSON='gt_segments_'$GENDER'.json'
CORREFORMER='none'
SORT_MODEL='none'
LOGDIR='./log/baselines_fps/dfaust_'$GENDER'_'$MODEL'_f'$N_FRAMES'_p'$N_POINTS'_shuffle_'$POINTS_SHUFFLE'_aug'$AUGMENT'_b'$BATCH_SIZE'_u'$STEPS_PER_UPDATE'_sort.'$SORT_MODEL'/'

taskset -c 65-129 python3 train_action_pred.py --dataset_path $DATASET_PATH --pc_model $MODEL --steps_per_update $STEPS_PER_UPDATE --frames_per_clip $N_FRAMES --batch_size $BATCH_SIZE --shuffle_points $POINTS_SHUFFLE --logdir $LOGDIR --n_epochs $N_EPOCHS --n_points $N_POINTS --sampler $SAMPLER --data_augmentation $AUGMENT --gender $GENDER --correformer $CORREFORMER --sort_model $SORT_MODEL
taskset -c 65-129 python3 test_action_pred.py --dataset_path $DATASET_PATH --pc_model $MODEL --frames_per_clip $N_FRAMES --batch_size $TEST_BATCH_SIZE --shuffle_points $POINTS_SHUFFLE --n_points $N_POINTS --model $(printf %06d $TEST_ITER).pt --model_path $LOGDIR --set $SET --gender $GENDER --correformer $CORREFORMER --sort_model $SORT_MODEL
taskset -c 65-129 python3 ./evaluation/evaluate.py --results_path $LOGDIR'results_'$(printf %06d $TEST_ITER)'/' --dataset_path $DATASET_PATH --set $SET --gt_segments_json_filename $GT_JSON --gender $GENDER