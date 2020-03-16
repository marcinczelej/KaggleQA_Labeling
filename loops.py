import os
import time

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
import horovod.tensorflow as hvd

from datetime import timedelta

from parameters import *
from utilities import accumulated_gradients
from metric import *
from utilities import CustomSchedule, CosineDecayWithWarmup

def train_loop(model, loss_fn, metric, train_ds, test_ds, checkpoint_dir, train_elements, test_elements):
    """
        Method that contains training loop for given epochs
        
        Parameters:
            model - model that should be trained
            loss_fn - loss function that sohuld be used during this training
            metric - metric that should be used to measure performance
            train_ds - dataset that should be used during training
            test_ds - test dataset that should be used for testing after each epoch training
            checkpoint_dir - directory there to save checkpoint
    """
    
    # splitting trainalble variables to two tables
    trainable = model.trainable_variables
    backbone_variables = []
    head_variables = []
    for var in trainable:
        if "backbone" in var.name:
            backbone_variables.append(var)
        else:
            head_variables.append(var)
    
    assert(len(trainable) == (len(backbone_variables) + len(head_variables)))

    @tf.function
    def train_step(inputs, y_true, first_batch):
        with tf.GradientTape() as tape:
            ids_mask, type_ids_mask, attention_mask = inputs[:, 0, :], inputs[:, 1, :], inputs[:, 2, :]
            y_pred = model(ids_mask, 
                           attention_mask= attention_mask, 
                           token_type_ids=type_ids_mask, 
                           training=True)
            loss = loss_fn(y_true, y_pred)
        
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss, trainable)

        return loss, grads, tf.math.sigmoid(y_pred)

    @tf.function
    def test_step(inputs, y_true):
        ids_mask, type_ids_mask, attention_mask = inputs[:, 0, :], inputs[:, 1, :], inputs[:, 2, :]
        y_pred = model(ids_mask, 
                       attention_mask= attention_mask, 
                       token_type_ids=type_ids_mask, 
                       training=False)

        loss = loss_fn(y_true, y_pred)
        return loss, tf.math.sigmoid(y_pred)

    last_loss = -999999
    
    bert_cosine_scheduler = CosineDecayWithWarmup(warmup_steps=warmup_steps, 
                                             total_steps=epochs * (train_elements//batch_size) // gradient_accumulate_steps, 
                                             base_lr=lr*hvd.size())
    
    head_cosine_scheduler = CosineDecayWithWarmup(warmup_steps=warmup_steps, 
                                             total_steps=epochs * (train_elements//batch_size) // gradient_accumulate_steps, 
                                             base_lr=lr*500*hvd.size())

    backbone_optimizer = tf.optimizers.Adam(learning_rate = bert_cosine_scheduler)
    head_optimizer = tf.optimizers.Adam(learning_rate = head_cosine_scheduler)
    
    for epoch in range(epochs):
        start = time.time()
        gradients = None
        train_losses = 0.0
        test_losses = 0.0
        train_preds = []
        test_preds = []
        train_targets = []
        test_targets = []
        global_batch = 0
        for batch_nr, (inputs, y_true) in enumerate(train_ds):
            loss, current_gradient, y_pred = train_step(inputs, y_true, batch_nr==0)
            train_losses += loss/((train_elements//batch_size) * gradient_accumulate_steps)
            train_preds.append(y_pred)
            train_targets.append(y_true)
            gradients = accumulated_gradients(gradients, current_gradient, gradient_accumulate_steps)

            if (batch_nr +1)%gradient_accumulate_steps ==0:
                """print(np.asarray(gradients).shape)"""
                backbone_gradients = gradients[:len(backbone_variables)]
                head_gradients = gradients[len(backbone_variables):]
                
                """print("backbone grad len ", len(backbone_gradients))
                print("head grad len     ", len(head_gradients))
                print("backbone var len ", len(backbone_variables))
                print("head var len     ", len(head_variables))"""
                
                
                bckbone_op = backbone_optimizer.apply_gradients(zip(backbone_gradients, backbone_variables))
                head_op = head_optimizer.apply_gradients(zip(head_gradients, head_variables))
                tf.group(bckbone_op, head_op)
                
                #optimizer.apply_gradients(zip(gradients, trainable))
                global_batch +=1
                gradients = None

                if batch_nr == 0:
                    print("first batch")
                    hvd.broadcast_variables(trainable, root_rank=0)
                    hvd.broadcast_variables(backbone_optimizer.variables(), root_rank=0)
                    hvd.broadcast_variables(head_optimizer.variables(), root_rank=0)

            """if batch_nr % 100 == 0 and hvd.local_rank() == 0:
                print('Step {} loss {}'  .format(batch_nr, loss, ))"""

        for _, (inputs, y_true) in enumerate(test_ds):
            loss, y_pred = test_step(inputs, y_true)
            test_losses += loss/((test_elements//batch_size) * gradient_accumulate_steps)
            test_preds.append(y_pred)
            test_targets.append(y_true)

        test_metric = metrics[metric](test_targets, test_preds)
        train_metric = metrics[metric](train_targets, train_preds)
        
        elapsed = (time.time() - start)
        if hvd.rank() == 0:
            print("epoch {}/{} train loss {} test loss {} test metric {} train metric {} epoch time {}" \
                  .format(epoch+1, epochs, train_losses, test_losses, test_metric, train_metric, str(timedelta(seconds=elapsed))))

        if test_metric > last_loss:
            if hvd.rank() == 0:
                model.save_pretrained(checkpoint_dir)
                last_loss = test_metric
                print("model for {} saved under {}... " .format(model.getName(), checkpoint_dir))
        
def create_pseudo_labels_loop(model, pseudo_labeling_ds, pseudo_labeling_df, fold_nr):
    """
        Method responsible for pseudo labeling data
        
        Parameters:
            model - nn model that will be used for prediction
            pseudo_labeling_ds - dataset with inputs for prediction are stored (.csv file)
            pseudo_labeling_df - dataframe with preprocessed input to be pseudolabeled. Preprocessing should be done with preprocessing.dataPreprocessor class
            fold_nr - fold number
            
        Return:
            predicted_df - Dataframe with predictions
        
        Pseudolabeling informations: 
            https://datawhatnow.com/pseudo-labeling-semi-supervised-learning/
            
        Pseudolables are saved into ./dataframes/pseudo_labeled_MODEL_NAME_fold-FOLD_NR.csv
    """

    pseudo_predictions = []
    for _, inputs in enumerate(pseudo_labeling_ds):
        ids_mask, type_ids_mask, attention_mask = inputs[:, 0, :], inputs[:, 1, :], inputs[:, 2, :]
        predicted = model(ids_mask, 
                      attention_mask= attention_mask, 
                      token_type_ids=type_ids_mask, 
                      training=False)

        pseudo_predictions.extend(predicted.numpy())

    print("predicting pseudo-labels done ...")
    pseudo_predictions = tf.math.sigmoid(pseudo_predictions)
    predicted_df = pseudo_labeling_df.copy(deep=True)
    for idx, col in enumerate(target_columns):
        predicted_df[col] = pseudo_predictions[:, idx]
    
    return predicted_df