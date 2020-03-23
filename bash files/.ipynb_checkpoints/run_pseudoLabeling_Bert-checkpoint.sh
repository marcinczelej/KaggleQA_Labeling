#!/bin/bash

clear

horovodrun -np 4 --timeline-filename ../timeline.json python -W ignore ../src/QA_labeling.py --checkpoint_dir="../checkpoints/Bert_1" --csv_dir="../dataframes/Bert_1" --model_name="BertForQALabeling" --mode="Pseudo_labeling"
