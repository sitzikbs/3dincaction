
DATASET_PATH='/home/sitzikbs/Datasets/dfaust/'
#DATASET_PATH='/data1/datasets/dfaust/'
MODEL='pn1'
STEPS_PER_UPDATE=20
N_FRAMES=32
BATCH_SIZE=8

N_POINTS=1024
N_EPOCHS=30
POINTS_SHUFFLE='once'
LOGDIR='./log/'$MODEL'_f'$N_FRAMES'_p'$N_POINTS'_shuffle_'$POINTS_SHUFFLE'/'
SET='test'

python3 train_action_pred.py --dataset_path $DATASET_PATH --pc_model $MODEL --steps_per_update $STEPS_PER_UPDATE --frames_per_clip $N_FRAMES --batch_size $BATCH_SIZE --shuffle_points $POINTS_SHUFFLE --logdir $LOGDIR --n_epochs $N_EPOCHS --n_points $N_POINTS
python3 test_action_pred.py --dataset_path $DATASET_PATH --pc_model $MODEL --frames_per_clip $N_FRAMES --batch_size $BATCH_SIZE --shuffle_points $POINTS_SHUFFLE --n_points $N_POINTS --model '000005.pt' --model_path $LOGDIR --set $SET
python3 ./evaluation/evaluate.py --results_path $LOGDIR'/results/' --dataset_path $DATASET_PATH --set $SET