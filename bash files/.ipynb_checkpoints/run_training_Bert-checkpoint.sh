#!/bin/bash

clear

resume_training=${1:-False}

if [ $resume_training = 'continue' ]; then
    echo "Resuming training"
    continue_training="continue"
else
    continue_training="from_scratch"
    echo "Running from scratch"
fi

horovodrun -np 4 --timeline-filename ../timeline.json python -W ignore ../src/QA_labeling.py --checkpoint_dir="../checkpoints/Bert_1" --csv_dir="../dataframes/Bert_1" --model_name="BertForQALabeling" --continue_training=$continue_training --mode="Train"