import os
from pathlib import Path

import tensorflow as tf
import pandas as pd

from sklearn.model_selection import KFold
import horovod.tensorflow as hvd

from parameters import *
from loops import *

from transformers import BertTokenizer, RobertaTokenizer

from preprocessing import dataPreprocessor
from BertModel import BertForQALabeling
from RoBERTaModel import *

models = {"RoBERTaForQALabeling": (RoBERTaForQALabeling, "roberta-base", "Roberta", RobertaTokenizer),
          "RoBERTaForQALabelingMultipleHeads": (RoBERTaForQALabelingMultipleHeads, "roberta-base", "Roberta", RobertaTokenizer),
          "BertForQALabeling": (BertForQALabeling, "bert-base-uncased", "Bert", BertTokenizer)}

class Trainer(object):
    @classmethod
    def prepare_and_save_folds(cls, input_df, model_name):
        
        print("preparing fold csv file as training is done from beginning")
        
        kf = KFold(Params.n_splits)
        folded_data = pd.DataFrame(data=None, columns=input_df.columns)
        
        for fold_nr, (train_idx, test_idx) in enumerate(kf.split(input_df)):
            # get data from dataframe
            train_df = pd.DataFrame(input_df.iloc[train_idx])
            test_df = pd.DataFrame(input_df.iloc[test_idx])
            
            train_df["fold_nr"] = fold_nr
            test_df["fold_nr"] = fold_nr
            train_df["type"] = "train"
            test_df["type"] = "test"
            folded_data = pd.concat([folded_data, train_df, test_df], ignore_index=True)
            
        Path(Params.csv_save_dir).mkdir(parents=True, exist_ok=True)
        folded_data.to_csv(os.path.join(Params.csv_save_dir, "{}-kfold5.csv" .format(model_name)))
        print("fold csv preparing DONE")
    
    @classmethod
    def create_dataset(cls, input_df):
        q_title = input_df['question_title'].values
        q_body = input_df['question_body'].values
        answer = input_df['answer'].values
        y_true = input_df[target_columns].to_numpy()

        ds = tf.data.Dataset.from_tensor_slices((q_title, q_body, answer, y_true)). \
                        shuffle(input_df.shape[0]//4, reshuffle_each_iteration=True). \
                        batch(batch_size=Params.batch_size, drop_remainder=False)
        return ds
    
    @classmethod
    def train(cls, model_name, input_df):
        
        """
            Setting up tokenizer
            - if training from scratch thern load default one and save it into given directory
            - if loading one, load ift form directory
            
            Setting preprocessing variables:
            - logger enabled
            - select what model to use
            - setting up tokenizer
            - preprocessing tokenizer vocab
        """
        checkpoint_dir = os.path.join(Params.save_dir, "{}_tokenizer_data" .format(model_name))

        if Params.resume_training == False:
            cls.prepare_and_save_folds(input_df, model_name)
            tokenizer = models[model_name][3].from_pretrained(models[model_name][1])
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            print("saving tokenizer in ", checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
        else:
            tokenizer = models[model_name][3].from_pretrained(checkpoint_dir)
        
        # prepare tokenizer vocabulary without any special tags
        # set dataPreprocessor varaibles
        dataPreprocessor.logger = False
        dataPreprocessor.model = models[model_name][2]
        dataPreprocessor.tokenizer = tokenizer
        dataPreprocessor.preprocess_vocab()
        
        folded_data_df = pd.read_csv(os.path.join(Params.csv_save_dir, "{}-kfold5.csv" .format(model_name)))
        
        for fold_nr in range(Params.starting_fold, Params.n_splits):
            print("Training Fold {}/{} " .format(fold_nr+1, Params.n_splits))
            
            # get data from dataframe
            train_df = folded_data_df.loc[(folded_data_df["fold_nr"] == fold_nr) & (folded_data_df["type"] == "train")]
            test_df = folded_data_df.loc[(folded_data_df["fold_nr"] == fold_nr) & (folded_data_df["type"] == "test")]
            
            # create train and test datasets
            train_ds = cls.create_dataset(train_df)
            test_ds = cls.create_dataset(test_df)

            bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            
            checkpoint_dir = os.path.join(Params.save_dir, '{}_fold_{}' .format(model_name, fold_nr))
            
            # if starting from scratch 
            # - load default model weights
            # - create directory to save trained weights
            if Params.resume_training == False:
                print("training from scratch...")
                weights = models[model_name][1]
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            else:
                # only load checkpoints for given fold. next ones are not started yet !!!!
                print("loading weights from {}..." .format(checkpoint_dir))
                weights = checkpoint_dir
                
            model = models[model_name][0].from_pretrained(weights, 
                                                          num_labels=Params.num_labels, 
                                                          output_hidden_states=True)
            print("QQQQQQQQQQQQQQQ start training")
            train_loop(model=model,
                       loss_fn=bce_loss, 
                       metric="spearmanr", 
                       train_ds=train_ds, 
                       test_ds=test_ds,  
                       checkpoint_dir=checkpoint_dir,
                       train_elements = train_df.shape[0],
                       test_elements=test_df.shape[0], 
                       fold_nr=fold_nr, 
                       model_name=model_name)
            
            Params.resume_training=False
            
    @classmethod
    def pseudo_predict(cls, model_name, pseudo_df):
        
        all_dataframes = []
        
        for fold_nr in range(Params.n_splits):
            print("Pseudo labeling Fold {}/{} " .format(fold_nr+1, Params.n_splits))
            
            """    
                Pseudo labeling for given fold using stackexchange data

                    1. restoring best checkpoint for given fold
                    2. predicting output values for stackexchange
                    3. saving predicted data into csv file 
            """
            
            checkpoint_dir = os.path.join(Params.save_dir, '{}_fold_{}' .format(model_name, fold_nr))
    
            model = models[model_name][0].from_pretrained(checkpoint_dir)
            print("best checkpoint for fold {} restored from {} ..." .format(fold_nr+1, checkpoint_dir))
            
            checkpoint_dir = os.path.join(Params.save_dir, "{}_tokenizer_data" .format(model_name))
            tokenizer = models[model_name][3].from_pretrained(checkpoint_dir)
            dataPreprocessor.tokenizer = tokenizer
            dataPreprocessor.model = models[model_name][2]
            
            print("creating pseudo-labels...")
            
            #preprocessing of train dataframe
            q_title = pseudo_df['question_title'].values
            q_body = pseudo_df['question_body'].values
            answer = pseudo_df['answer'].values
            pseudo_input = dataPreprocessor.preprocessBatch(q_body, q_title, answer, max_seq_lengths=(26,260,210,500))
            
            pseudo_labeling_ds = tf.data.Dataset.from_tensor_slices((pseudo_input)).batch(batch_size=Params.batch_size, drop_remainder=False)
            
            pseudo_labels_df = create_pseudo_labels_loop(model=model,  
                                                         pseudo_labeling_ds=pseudo_labeling_ds,
                                                         pseudo_labeling_df=pseudo_df,
                                                         fold_nr=fold_nr)
            pseudo_labels_df["fold_nr"] = fold_nr
            
            file_name = "pseudo_labeled_{}_fold-{}.csv" .format(model_name, fold_nr)
            
            pseudo_labels_df.to_csv(os.path.join(Params.csv_save_dir, file_name), index=False)
            all_dataframes.append(pseudo_labels_df)
            fold_nr += 1
            
            