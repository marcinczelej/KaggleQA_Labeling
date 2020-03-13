import os
from pathlib import Path

import tensorflow as tf
import pandas as pd

from sklearn.model_selection import KFold
import horovod.tensorflow as hvd

from parameters import *
from utilities import CustomSchedule
from loops import *

from BertModel import BertForQALabeling
from RoBERTaModel import *

models = {"RoBERTaForQALabeling": (RoBERTaForQALabeling, "roberta-base"),
          "RoBERTaForQALabelingMultipleHeads": (RoBERTaForQALabelingMultipleHeads, "roberta-base"),
          "BertForQALabeling": (BertForQALabeling, "bert-base-uncased")}

class Trainer(object):
    @classmethod
    def train(cls, model_name, preprocessedInput, targets, preprocessedPseudo, weights_directory=None):

        kf = KFold(n_splits)
        fold_nr =0

        for train_idx, test_idx in kf.split(preprocessedInput):
            print("Fold{}/{} " .format(fold_nr, n_splits))
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
            
            if weights_directory == None:
                weights = models[model_name][1]
            else:
                weights = weights_directory
                
            model = models[model_name][0].from_pretrained(weights, num_labels=num_labels, output_hidden_states=True)
            
            print("loaded model ", model.getName())

            checkpoint_dir = os.path.join(save_dir, '{}_fold_{}' .format(model.getName(), fold_nr))
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

            train_loop(model=model,
                       optimizer=optimizer, 
                       loss_fn=bce_loss, 
                       metric="spearmanr", 
                       train_ds=train_ds, 
                       test_ds=test_ds,  
                       checkpoint_dir=checkpoint_dir)
            fold_nr += 1
            
    @classmethod
    def pseudo_predict(cls, model_name, preprocessedPseudo, pseudo_df):
        
        for fold_nr in range(n_splits):
            print("Fold{}/{} " .format(fold_nr, n_splits))
            
            """    
                Pseudo labeling for given fold using stackexchange data

                    1. restoring best checkpoint for given fold
                    2. predicting output values for stackexchange
                    3. saving predicted data into csv file 
            """
            
            checkpoint_dir = os.path.join(save_dir, '{}_fold_{}' .format(model_name, fold_nr))
    
            model = models[model_name][0].from_pretrained(checkpoint_dir)
            print("best checkpoint for fold {} restored from {} ..." .format(fold_nr, checkpoint_dir))
            
            print("creating pseudo-labels...")
            pseudo_labeling_ds = tf.data.Dataset.from_tensor_slices((preprocessedPseudo)).batch(batch_size=batch_size, drop_remainder=True)
            
            pseudo_labels_df = create_pseudo_labels_loop(model=model,  
                                                         pseudo_labeling_ds=pseudo_labeling_ds,
                                                         pseudo_labeling_df=pseudo_df,
                                                         fold_nr=fold_nr)
            
            df_save_dir = os.path.join(csv_save_dir, model.getName())
            file_name = "pseudo_labeled_{}_fold-{}.csv" .format(model.getName(), fold_nr)
            Path(df_save_dir).mkdir(parents=True, exist_ok=True)
            
            pseudo_labels_df.to_csv(os.path.join(df_save_dir, file_name), index=False)
            fold_nr += 1
            
            