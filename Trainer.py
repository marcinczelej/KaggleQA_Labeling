import os
from pathlib import Path

import tensorflow as tf
import pandas as pd

from sklearn.model_selection import KFold
import horovod.tensorflow as hvd

from preprocessing import dataPreprocessor
from parameters import *
from loops import *

from BertModel import BertForQALabeling
from RoBERTaModel import *

import logging
logging.basicConfig(level=logging.ERROR)

models = {"RoBERTaForQALabeling": (RoBERTaForQALabeling, "roberta-base"),
          "RoBERTaForQALabelingMultipleHeads": (RoBERTaForQALabelingMultipleHeads, "roberta-base"),
          "BertForQALabeling": (BertForQALabeling, "bert-base-uncased")}

class Trainer(object):
    @classmethod
    def train(cls, model_name, tokenizer, input_df, weights_directory=None):
        """
            Change to passing df not preprocessed input
            Do preprocessing here
        """

        kf = KFold(n_splits)
        folded_data = pd.DataFrame(data=None, columns=input_df.columns)
        
        for fold_nr, (train_idx, test_idx) in enumerate(kf.split(input_df)):
            print("Fold{}/{} " .format(fold_nr+1, n_splits))
            
            # get data from dataframe
            train_df = pd.DataFrame(input_df.iloc[train_idx])
            test_df = pd.DataFrame(input_df.iloc[test_idx])
            
            train_df["fold_nr"] = fold_nr
            test_df["fold_nr"] = fold_nr
            train_df["part"] = "train"
            test_df["part"] = "test"
            folded_data = pd.concat([folded_data, train_df, test_df], ignore_index=True)

            #preprocessing of train dataframe
            q_title = train_df['question_title'].values
            q_body = train_df['question_body'].values
            answer = train_df['answer'].values
            train_input = dataPreprocessor.preprocessBatch(q_body, q_title, answer, max_seq_lengths=(26,260,210,500))
            train_target = train_df[target_columns].to_numpy()
            
            #preprocessing of test dataframe
            q_title = test_df['question_title'].values
            q_body = test_df['question_body'].values
            answer = test_df['answer'].values
            test_input = dataPreprocessor.preprocessBatch(q_body, q_title, answer, max_seq_lengths=(26,260,210,500))
            test_target = test_df[target_columns].to_numpy()

            #train dataset
            train_ds = tf.data.Dataset.from_tensor_slices((train_input, train_target)). \
                                     shuffle(len(train_input)//4, reshuffle_each_iteration=True). \
                                     batch(batch_size=batch_size, drop_remainder=False)

            #test dataset
            test_ds = tf.data.Dataset.from_tensor_slices((test_input, test_target)). \
                                     shuffle(len(test_input)//4, reshuffle_each_iteration=True). \
                                     batch(batch_size=batch_size, drop_remainder=False)

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
                       loss_fn=bce_loss, 
                       metric="spearmanr", 
                       train_ds=train_ds, 
                       test_ds=test_ds,  
                       checkpoint_dir=checkpoint_dir,
                       elements=len(test_idx))
        
        Path(csv_save_dir).mkdir(parents=True, exist_ok=True)
        folded_data.to_csv(os.path.join(csv_save_dir, "{}-kfold5.csv" .format(model.getName())))
            
    @classmethod
    def pseudo_predict(cls, model_name, pseudo_df):
        
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
            
            #preprocessing of train dataframe
            q_title = pseudo_df['question_title'].values
            q_body = pseudo_df['question_body'].values
            answer = pseudo_df['answer'].values
            pseudo_input = dataPreprocessor.preprocessBatch(q_body, q_title, answer, max_seq_lengths=(26,260,210,500))
            
            pseudo_labeling_ds = tf.data.Dataset.from_tensor_slices((pseudo_input)).batch(batch_size=batch_size, drop_remainder=False)
            
            pseudo_labels_df = create_pseudo_labels_loop(model=model,  
                                                         pseudo_labeling_ds=pseudo_labeling_ds,
                                                         pseudo_labeling_df=pseudo_df,
                                                         fold_nr=fold_nr)
            pseudo_labels_df["fold_nr"] = fold_nr
            
            df_save_dir = os.path.join(csv_save_dir, model.getName())
            file_name = "pseudo_labeled_{}_fold-{}.csv" .format(model.getName(), fold_nr)
            Path(df_save_dir).mkdir(parents=True, exist_ok=True)
            
            pseudo_labels_df.to_csv(os.path.join(df_save_dir, file_name), index=False)
            fold_nr += 1
            
            