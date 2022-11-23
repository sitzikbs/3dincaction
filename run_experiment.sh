
DATASET_PATH='/home/sitzikbs/Datasets/dfaust/'

MODEL='pn1_4d'
STEPS_PER_UPDATE=20
N_FRAMES=32
BATCH_SIZE=8
N_POINTS=1024
N_EPOCHS=30
POINTS_SHUFFLE='shuffle_once'
LOGDIR='./log/'$MODEL'_f'$N_FRAMES'_p'$N_PINTS'_'$POINTS_SHUFFLE/


python3 train_action_pred.py --pc_model $MODEL --steps_per_update $STEPS_PER_UPDATE --frames_per_clip $N_FRAMES --batch_size $BATCH_SIZE --shuffle_points $POINTS_SHUFFLE --logdir $LOGDOR --n_epochs $N_EPOCHS --n_points $N_POINTS
#python3 test_action_pred.py
#python3 ./evaluation/evaluate.py