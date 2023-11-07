import numpy as np
from ConvNet import ConvNet
from ResNet import ResNet
from MLPNet import MLPNet
import tensorflow as tf
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import keras.backend as K
import os
import random
import time
import joblib


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  tf.random.set_seed(seed)
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  os.environ["PYTHONHASHSEED"] = str(seed)


def print_best_callback(study, trial, study_file):
    print()
    print('trail number =', len(study.trials))
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
    print('saving study in to file', study_file)
    joblib.dump(study, study_file)
    

def objective(trial, xdata, ydata, weights, epochs, patience_epochs, study_file, seed):
    
    @tf.function
    def mae_func(y_true, y_pred):
        return K.mean(K.abs(y_true - y_pred))


    @tf.function
    def nll(y_true, y_pred, weights):
        nll = -y_pred.log_prob(y_true)
        nll *= weights
        nll = K.sum(nll, axis=-1)
        nll = K.mean(nll)
        return nll

    @tf.function
    def train_model(x, y, w):
            with tf.GradientTape() as tape:                
                y_pred = ml_model(x, training=True)
                loss = nll(tf.cast(y, dtype=type_casting), y_pred, tf.cast(w, dtype=type_casting) )     
            grads = tape.gradient(loss, ml_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, ml_model.trainable_weights))
        
    @tf.function
    def evaluate_model(x, y, w):
            y_pred = ml_model(x, training=False)
            loss_nll = nll(tf.cast(y, dtype=type_casting), y_pred,  tf.cast(w, dtype=type_casting))            
            mae.update_state(mae_func(tf.cast(y, dtype=type_casting), y_pred.mean()))
            nll_sum.update_state(loss_nll)

    
    type_casting = tf.float32
    Npixels = xdata.shape[1]
    Nnodes = Npixels
    Ntotal = xdata.shape[0]
    Ntrain = np.int32(Ntotal*0.8)
    Ntest = Ntotal - Ntrain
    xdata = np.expand_dims(xdata, axis=2)
     
    train_data = tf.data.Dataset.from_tensor_slices((xdata[:Ntrain], 
                                                     ydata[:Ntrain], 
                                                     weights[:Ntrain]))
    test_data = tf.data.Dataset.from_tensor_slices((xdata[Ntrain:], 
                                                    ydata[Ntrain:], 
                                                    weights[Ntrain:]))
    
    #choose hyper parameters for model training
    initial_learning_rate = trial.suggest_float('lr', 1e-4, 0.5, log=True)
    #powers of two
    batch_size = np.power(2, trial.suggest_int("batch_size", 5, 10))
    #four stages, inspired by ResNet
    num_blocks = np.full(trial.suggest_int("num_blocks", 1, 2), 1, dtype=np.int32)
    
    features_per_block = np.ones(num_blocks.shape, dtype=np.int32)
    layers_per_block = np.ones(num_blocks.shape, dtype=np.int32)
    #the features at first stage keeping feature as power of two
    features_per_block[0] = np.power(2, trial.suggest_int("features_per_block1", 2, 5))
    layers_per_block[0] = trial.suggest_int("num_layers1", 1, 2)
    
    #layers at each stage, maximum
    max_layers_per_block = 4

    for ci in range(1,len(num_blocks)):
        features_per_block[ci] = trial.suggest_int("features_per_block"+str(ci+1), 
                                             features_per_block[ci-1], 
                                             features_per_block[ci-1]*2, 
                                             features_per_block[ci-1])

        #each stage choice between number of layers in 
        #previous stage and up to maximum value
        layers_per_block[ci] = trial.suggest_int("layers_per_block"+str(ci+1), 
                                             layers_per_block[ci-1], 
                                             max_layers_per_block)

    
    #suggest a network
    network = trial.suggest_categorical("network", ["ConvNet", "ResNet"])
    
    print('')
    print('')
    if 'ResNet' in network:
        ml_model  = ResNet(layers_per_block, features_per_block, Nnodes, seed)
        print(network, layers_per_block, features_per_block)

    elif 'ConvNet' in network:
        ml_model  = ConvNet(layers_per_block, features_per_block, Nnodes, seed)
        print(network, layers_per_block, features_per_block) 

    optimizer = tf.keras.optimizers.Adam(initial_learning_rate)
    ml_model.build(input_shape=(None, xdata.shape[1], 1))
    ml_model.compile(optimizer=optimizer)


    best_metric = np.Infinity
    current_metric = np.Infinity
        
    mae = tf.keras.metrics.Mean()
    nll_sum = tf.keras.metrics.Sum()
    
    train_data = train_data.shuffle(Ntrain).batch(batch_size)
    test_data = test_data.shuffle(Ntest).batch(batch_size)
    Nbatches = np.float32(Ntest/batch_size)
    
    print('number of batches =', Nbatches)

    no_improvement_count = 0
     
    for epoch in range(epochs):     
        start =  time.time()
        
        for step, (x_batch, y_batch, w_batch) in enumerate(train_data):
             train_model(x_batch, y_batch, w_batch)

        nll_sum.reset_state()
        mae.reset_state()
        
        for step, (x_batch, y_batch, w_batch) in enumerate(test_data):
            evaluate_model(x_batch, y_batch, w_batch)

        current_metric = nll_sum.result().numpy()
        
        if current_metric < best_metric:
            no_improvement_count = 0
            best_metric = current_metric
        else:
            no_improvement_count += 1
        if no_improvement_count >= patience_epochs:
            break
        
        end =  time.time()
        
        print('Epoch', epoch+1, np.round(end-start, 1), '[sec]', 
              "{:f}".format(nll_sum.result().numpy()/Npixels/Nbatches), 
              "{:f}".format(mae.result().numpy()), no_improvement_count )
        
        nll_sum.reset_state()
        mae.reset_state()

    return best_metric
