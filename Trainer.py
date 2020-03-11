import os

import tensorflow as tf
import pandas as pd

from sklearn.model_selection import KFold
import horovod.tensorflow as hvd

from parameters import *
from utilities import CustomSchedule
from loops import *

class Trainer(object):
    @classmethod
    def train(cls, model, tokenizer, preprocessedInput, targets, preprocessedPseudo):

        kf = KFold(n_splits)
        fold_nr =0

        for train_idx, test_idx in kf.split(preprocessedInput):

            # train test indices
            train_input = tf.gather(preprocessedInput, train_idx, axis=0)
            train_target = tf.gather(targets, train_idx, axis=0)

            test_input = tf.gather(preprocessedInput, test_idx, axis=0)
            test_target = tf.gather(targets, test_idx, axis=0)

            #train dataset
            train_ds = tf.data.Dataset.from_tensor_slices((train_input, train_target)). \
                                     shuffle(len(train_input)//4, reshuffle_each_iteration=True). \
                                     batch(batch_size=batch_size, drop_remainder=True)

            #test dataset
            test_ds = tf.data.Dataset.from_tensor_slices((test_input, test_target)). \
                                     shuffle(len(test_input)//4, reshuffle_each_iteration=True). \
                                     batch(batch_size=batch_size, drop_remainder=True)

            lr_scheduler = CustomSchedule(warmup_steps=warmup_steps*2, 
                                          num_steps=targets.shape[0]//batch_size, 
                                          base_lr=lr*hvd.size())

            #optimizer = tfa.optimizers.AdamW(learning_rate=lr_scheduler, weight_decay=decay)
            optimizer = tf.optimizers.Adam(learning_rate = lr_scheduler)
            bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

            """model = BertForQALabeling.from_pretrained('bert-base-uncased', num_labels=num_labels, output_hidden_states=True)"""

            """model = RoBERTaForQALabeling.from_pretrained('roberta-base', num_labels=num_labels, output_hidden_states=True)"""

            """model = RoBERTaForQALabelingMultipleHeads.from_pretrained('roberta-base', num_labels=num_labels, output_hidden_states=True)"""

            checkpoint_dir = './checkpoints/best_{}_fold_{}' .format(model.getName(), fold_nr)
            checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

            train_loop(model=model, 
                       optimizer=optimizer, 
                       loss=bce_loss, 
                       metric="spearmanr", 
                       train_ds=train_ds, 
                       test_ds=test_ds, 
                       checkpoint=checkpoint, 
                       checkpoint_dir=checkpoint_dir)

            """    
                Pseudo labeling for given fold using stackexchange data

                    1. restoring best checkpoint for given fold
                    2. predicting output values for stackexchange
                    3. saving predicted data into csv file 
            """
            pseudo_labeling_ds = tf.data.Dataset.from_tensor_slices((preprocessedPseudo)).batch(batch_size=batch_size, drop_remainder=True)
            
            pseudo_labels_df = PseudoLabeler.create_pseudo_labels(checkpoint=checkpoint, 
                                                               model=model, 
                                                               checkpoint_dir=checkpoint_dir, 
                                                               pseudo_labeling_df=pseudo_labeling_df, 
                                                               fold_nr=fold_nr)
            pseudo_labels_df.to_csv(
                        os.path.join("./dataframes/pseudo_labeled_{}_fold-{}.csv" .format(model.getName(), fold_nr)), index=False)
            fold_nr += 1