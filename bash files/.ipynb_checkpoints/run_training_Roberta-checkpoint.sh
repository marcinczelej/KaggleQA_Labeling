#!/bin/bash

#!/bin/bash

clear

resume_training=${1:-False}
fold_nr=${2:-False}

if [ $resume_training = 'continue']; then
    echo "Resuming training"
    continue_training="continue"
    fold_nr=$2
else
    continue_training="from_scratch"
    echo "Running from scratch"
    fold_nr=0
fi

horovodrun -np 4 --timeline-filename ../timeline.json python -W ignore ../src/QA_labeling.py --checkpoint_dir="../checkpoints/Roberta_1" --csv_dir="../dataframes/Roberta_1" --model_name="RoBERTaForQALabeling" --continue_training=$continue_training --mode="Train" --fold_nr=fold_nr