#DATASET_PATH='/home/sitzikbs/datasets/dfaust/'
DATASET_PATH='/home/sitzikbs/Datasets/dfaust/'

MODEL='pn1_4d'
STEPS_PER_UPDATE=2
N_FRAMES=64
BATCH_SIZE=8
TEST_BATCH_SIZE=1
TEST_ITER=200
GENDER='female'

N_POINTS=1024
N_EPOCHS=200
POINTS_SHUFFLE='each_frame'
SAMPLER='weighted'
AUGMENT=1

LOGDIR='./log/dfaust_'$GENDER'_'$MODEL'_f'$N_FRAMES'_p'$N_POINTS'_shuffle_'$POINTS_SHUFFLE'_aug'$AUGMENT'_b'$BATCH_SIZE'_u'$STEPS_PER_UPDATE'/'
SET='test'
GT_JSON='gt_segments_'$GENDER'.json'
CORREFORMER='none'

python3 train_action_pred.py --dataset_path $DATASET_PATH --pc_model $MODEL --steps_per_update $STEPS_PER_UPDATE --frames_per_clip $N_FRAMES --batch_size $BATCH_SIZE --shuffle_points $POINTS_SHUFFLE --logdir $LOGDIR --n_epochs $N_EPOCHS --n_points $N_POINTS --sampler $SAMPLER --data_augmentation $AUGMENT --gender $GENDER --correformer $CORREFORMER
python3 test_action_pred.py --dataset_path $DATASET_PATH --pc_model $MODEL --frames_per_clip $N_FRAMES --batch_size $TEST_BATCH_SIZE --shuffle_points $POINTS_SHUFFLE --n_points $N_POINTS --model $(printf %06d $TEST_ITER).pt --model_path $LOGDIR --set $SET --visualize_results 0 --gender $GENDER --correformer $CORREFORMER
python3 ./evaluation/evaluate.py --results_path $LOGDIR'results_'$(printf %06d $TEST_ITER)'/' --dataset_path $DATASET_PATH --set $SET --gt_segments_json_filename $GT_JSON