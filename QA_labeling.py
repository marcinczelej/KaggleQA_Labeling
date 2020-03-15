import os
import numpy as np

import pandas as pd
import tensorflow as tf

from transformers import BertTokenizer, RobertaTokenizer
from Trainer import Trainer
from preprocessing import dataPreprocessor
from parameters import *

from pathlib import Path

import horovod.tensorflow as hvd

tf.__version__
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

data_dir = "google-quest-challenge/"

train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
stack_df = pd.read_csv(os.path.join(data_dir, "stackexchange.csv"))
submit_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))    

input_df = train_df[train_columns + target_columns]

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

dataPreprocessor.logger = False
dataPreprocessor.tokenizer = tokenizer
dataPreprocessor.model = "Roberta"

model_name = "RoBERTaForQALabeling"

checkpoint_dir = os.path.join(save_dir, "{}_tokenizer_data" .format(model_name))
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
print("saving tokenizer in ", checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)

Trainer.train(model_name=model_name,
              tokenizer=tokenizer,
              input_df=input_df)

Trainer.pseudo_predict(model_name=model_name, 
                       pseudo_df=stack_df)