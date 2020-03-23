import os
import time

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
import horovod.tensorflow as hvd

from datetime import timedelta

from preprocessing import dataPreprocessor
from parameters import *
from utilities import accumulated_gradients
from metric import *
from utilities import CustomSchedule, CosineDecayWithWarmup

def train_loop(model, loss_fn, metric, train_ds, test_ds, checkpoint_dir, train_elements, test_elements, fold_nr, model_name):
    """
        Method that contains training loop for given epochs
        
        Parameters:
            model - model that should be trained
            loss_fn - loss function that sohuld be used during this training
            metric - metric that should be used to measure performance
            train_ds - dataset that should be used during training
            test_ds - test dataset that should be used for testing after each epoch training
            checkpoint_dir - directory there to save checkpoint
            fold_nr, fold numpber that is currently evaluated
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
    def train_step(inputs, y_true):
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
    start_epoch = 0
    
    train_total_steps = Params.epochs*(train_elements//Params.batch_size) // Params.gradient_accumulate_steps
    
    bert_cosine_scheduler = CosineDecayWithWarmup(warmup_steps=Params.warmup_steps, 
                                             total_steps=train_total_steps, 
                                             base_lr=Params.lr*hvd.size())
    
    head_cosine_scheduler = CosineDecayWithWarmup(warmup_steps=Params.warmup_steps, 
                                             total_steps=train_total_steps, 
                                             base_lr=Params.lr*500*hvd.size())

    backbone_optimizer = tf.optimizers.Adam(learning_rate = bert_cosine_scheduler)
    head_optimizer = tf.optimizers.Adam(learning_rate = head_cosine_scheduler)
    
    train_checkpoint_dir = os.path.join(Params.save_dir, '{}_fold_{}_ckpt' .format(model_name, fold_nr))
    
    train_checkpoint = tf.train.Checkpoint(current_epoch=tf.Variable(0), 
                                           backbone_optimizer=backbone_optimizer, 
                                           head_optimizer=head_optimizer)
    
    ckpt_manager = tf.train.CheckpointManager(train_checkpoint, 
                                              train_checkpoint_dir, 
                                              max_to_keep=1)
    
    if Params.resume_training == True:
        train_checkpoint.restore(ckpt_manager.latest_checkpoint)
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        start_epoch = int(train_checkpoint.current_epoch)
        print(start_epoch)
    
    for epoch in range(start_epoch, Params.epochs):
        start = time.time()
        gradients = None
        train_losses = 0.0
        test_losses = 0.0
        train_preds = []
        test_preds = []
        train_targets = []
        test_targets = []
        global_batch = 0
        
        """
                                               TRAINING LOOP
        """
        for batch_nr, (q_title, q_body, answer, y_true) in enumerate(train_ds):
            train_input = dataPreprocessor.preprocessBatch(q_body=q_body.numpy(), 
                                                           q_title=q_title.numpy(), 
                                                           answer=answer.numpy(), 
                                                           max_seq_lengths=(26,260,210,500), 
                                                           mask_words=True)
            
            loss, current_gradient, y_pred = train_step(inputs=train_input, 
                                                        y_true=y_true)
            
            train_losses += loss/((train_elements//Params.batch_size) * Params.gradient_accumulate_steps)
            train_preds.append(y_pred)
            train_targets.append(y_true)
            gradients = accumulated_gradients(gradients, current_gradient, Params.gradient_accumulate_steps)

            if (batch_nr +1)%Params.gradient_accumulate_steps ==0:
                backbone_gradients = gradients[:len(backbone_variables)]
                head_gradients = gradients[len(backbone_variables):]            
                
                bckbone_op = backbone_optimizer.apply_gradients(zip(backbone_gradients, backbone_variables))
                head_op = head_optimizer.apply_gradients(zip(head_gradients, head_variables))
                tf.group(bckbone_op, head_op)
                
                global_batch +=1
                gradients = None

                if batch_nr == 0:
                    print("first batch")
                    hvd.broadcast_variables(trainable, root_rank=0)
                    hvd.broadcast_variables(backbone_optimizer.variables(), root_rank=0)
                    hvd.broadcast_variables(head_optimizer.variables(), root_rank=0)

        """
                                               TESTING LOOP
        """
        for _, (q_title, q_body, answer, y_true) in enumerate(test_ds):
            test_input = dataPreprocessor.preprocessBatch(q_body=q_body.numpy(), 
                                                          q_title=q_title.numpy(), 
                                                          answer=answer.numpy(), 
                                                          max_seq_lengths=(26,260,210,500), 
                                                          mask_words=False)
            loss, y_pred = test_step(inputs=test_input, 
                                     y_true=y_true)
            
            test_losses += loss/((test_elements//Params.batch_size) * Params.gradient_accumulate_steps)
            test_preds.append(y_pred)
            test_targets.append(y_true)

        """
                                               METRICS COLLECTION
        """
        test_metric = metrics[metric](test_targets, test_preds)
        train_metric = metrics[metric](train_targets, train_preds)
        
        elapsed = (time.time() - start)
        if hvd.rank() == 0:
            print("\nFold {} epoch {}/{} train loss {} test loss {} test metric {} train metric {} epoch time {}" \
                  .format(fold_nr+1, 
                          epoch+1, 
                          Params.epochs, 
                          train_losses, 
                          test_losses, 
                          test_metric, 
                          train_metric, 
                          str(timedelta(seconds=elapsed))))

        """
                                               BEST MODEL SAVE
        """
        if test_metric > last_loss:
            if hvd.rank() == 0:
                model.save_pretrained(checkpoint_dir)
                last_loss = test_metric
                print("    model saved under {}... " .format(checkpoint_dir))

        if hvd.rank()==0:
            train_checkpoint.current_epoch.assign(epoch)
            ckpt_manager.save()
        
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