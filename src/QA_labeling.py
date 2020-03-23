import os
import argparse
import numpy as np

import pandas as pd
import tensorflow as tf

from Trainer import Trainer, models
from parameters import *

from pathlib import Path

import horovod.tensorflow as hvd
tf.__version__

import logging
logging.basicConfig(level=logging.ERROR)
def main(args):
    
    hvd.init()
    
    gpus = tf.config.list_physical_devices('GPU') 
    for gpu in gpus:
        print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print("gpus ", gpus)
        print("local rank ",hvd.local_rank())
        tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        print(tf.config.get_visible_devices())

    model_name = args.model_name
    Params.save_dir = args.checkpoint_dir
    Params.csv_save_dir=args.csv_dir
    Params.starting_fold = args.fold_nr
    
    if args.continue_training == "from_scratch":
        Params.resume_training = False
    elif args.continue_training == "continue":
        Params.resume_training = True
    
    options = vars(args)
    
    for key, val in vars(args).items():
        print("{} : {}" .format(key, val))

    data_dir = "../google-quest-challenge/"

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    stack_df = pd.read_csv(os.path.join(data_dir, "stackexchange.csv"))
    submit_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))    

    input_df = train_df[train_columns + target_columns]

    if args.mode == "Train":
        print("Training mode...")
        Trainer.train(model_name=model_name,
                      input_df=input_df[:30])
    
    elif args.mode == "Pseudo_labeling":
        print("Pseudo labeling mode...")
        Trainer.pseudo_predict(model_name=model_name, 
                               pseudo_df=stack_df)
    else:
        print("Prediction mode...")
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints/Roberta_1", required=True)
    parser.add_argument("--csv_dir", type=str, default="./dataframes/Roberta_1", required=True)
    parser.add_argument("--model_name", type=str, default="RoBERTaForQALabeling", required=True)
    parser.add_argument("--continue_training", type=str, default="from_scratch", help="possible modes from_scratch  |  continue")
    parser.add_argument("--mode", type=str, default="Train", help="mode of working : Train - training, Predict - prediction, Pseudo_labeling - predict pseudo labels", required=True)
    parser.add_argument("--fold_nr", type=int, default=0)

    args = parser.parse_args()

    main(args)