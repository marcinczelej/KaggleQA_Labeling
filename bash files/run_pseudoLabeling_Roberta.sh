#!/bin/bash

# !/bin/bash

clear

horovodrun -np 4 --timeline-filename ../timeline.json python -W ignore ../src/QA_labeling.py --checkpoint_dir="../checkpoints/Roberta_1" --csv_dir="../dataframes/Roberta_1" --model_name="RoBERTaForQALabeling" --mode="Pseudo_labeling"
