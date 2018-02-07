#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 10:10:55 2018

@author: David
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#%% Print program header
def print_header():
    print("==============================================================================================")
    print("description         :NN time series sandbox scripts.")
    print("author              :David Beam, github db4ai")
    print("date                :20110930")
    print("python version      :3.6.3 (v3.6.3:2c5fed8, Oct  3 2017, 18:11:49) [MSC v.1900 64 bit (AMD64)]")
    print("tensorflow version  :1.4.0")
    print("notes               :")
    print("==============================================================================================")


#%% Create NARX datasets
"""
    Creates a Nonlinear Auto-Regressive w/ Exogenous Input style dataset
    Defined by the number of delayed inputs and the number of delayed outptus
"""
def create_NARX_dataset(input, output, numDelayedInputs, numDelayedOutputs):
    # Calculate the sizes of the data
    numInputs = input.shape[1]
    numOutputs = output.shape[1]
    length = input.shape[0] - max(numDelayedInputs,numDelayedOutputs)
    width = ((numDelayedInputs + 1)*numInputs) + (numDelayedOutputs*numOutputs)
    
    # Placeholder to hold the dataset
    x_input_NARX = np.zeros((length, width) , dtype=np.float32)
    
    # Loop through all the inputs
    for i in range(max(0,max(numDelayedInputs+1,numDelayedOutputs)-1), input.shape[0]):
        
        # Append delayed inputs to the row        
        temp_row = input[i,:]
        for j in range(1,numDelayedInputs+1):
            temp_row = np.concatenate([temp_row, input[i-j]])
        
        # Append delayed outputs to the row
        for j in range(0,numDelayedOutputs):
            temp_row = np.concatenate([temp_row, output[i-j,:]], axis=0)
            
        x_input_NARX[i-max(numDelayedInputs+1,numDelayedOutputs),:] = temp_row
    return x_input_NARX


#%% Split the data into training and testing datasets
"""
    Takes the full dataset and splits it into two sets defined by the testing data size and the training data size
"""
def split_data(series,training,testing):
    testing  = series[-testing:]    #split off the testing data
    training = series[0:training]   #split off the training data
    return training,testing


#%% Split data into testing and training sets
"""
    Uses the split_data function to split the required datasets into training and testing sets
"""
def split_data_into_sets(x1,y_Realtime,y_Interval,y_Interval_Interpolated,x2,x2_Interval,trainingSamples,testingSamples,batchSize):
    molW_training,molW_test                                         = split_data(x1,trainingSamples,testingSamples)
    y_Realtime_training,y_Realtime_test                             = split_data(y_Realtime,trainingSamples,testingSamples)
    y_Interval_training,y_Interval_test                             = split_data(y_Interval,trainingSamples,testingSamples)
    y_Interval_Interpolated_training,y_Interval_Interpolated_test   = split_data(y_Interval_Interpolated,trainingSamples,testingSamples)
    x2_training,x2_test                                             = split_data(x2,trainingSamples,testingSamples)
    x2_Interval_training,x2_Interval_test                           = split_data(x2_Interval,trainingSamples,testingSamples)
    
    return molW_training,molW_test,y_Realtime_training,y_Realtime_test,y_Interval_training,y_Interval_test,y_Interval_Interpolated_training, \
        y_Interval_Interpolated_test,x2_training,x2_test,x2_Interval_training,x2_Interval_test


#%% Create datasets from the batches
"""
    Take a 2d array and reshapes it into a 3d array with the first dimension being the batch number
    [batch_size, time-step, sample]
"""
def make_batches(series,samples):
    data = series[:(len(series)-(len(series) % samples))]   #trim off extra to ensure equal size batches
    batches = data.reshape(-1, samples, series.shape[1])    #form batches
    return batches


#%% Import the data and separate into batches
def split_data_into_batches(x1,y_Realtime,y_Interval,y_Interval_Interpolated,x2,x2_Interval,trainingSamples,testingSamples,batchSize,numInputDelays,numOutputDelays):

    # Split the datasets into testing and training
    x1_training,x1_test, \
    y_Realtime_training,y_Realtime_test, \
    y_Interval_training,y_Interval_test, \
    y_Interval_Interpolated_training,y_Interval_Interpolated_test, \
    x2_training,x2_test, \
    x2_Interval_training,x2_Interval_test = \
        split_data_into_sets(x1,y_Realtime,y_Interval,y_Interval_Interpolated,x2,x2_Interval,trainingSamples,testingSamples,batchSize)
    
    # Create the input dataset for the NN model
    # This code uses only x2 as the input
    x_input = x2_training
    x_test  = x2_test
    # This code combines x1 and x2 into a single x array
    #x_input = np.concatenate((x1_training, x2_training), axis=1)
    #x_test  = np.concatenate((x1_test, x2_test), axis=1)
    
    # Create the input dataset for the NARX model
    x_input_NARX = create_NARX_dataset(x_input, y_Realtime_training,numInputDelays,numOutputDelays)
    
    # Create batches for the NN model
    x_input_batches                 = make_batches(x_input, batchSize)
    x_test_batches                  = make_batches(x_test, batchSize)
    x_input_NARX_batches            = make_batches(x_input, batchSize)
    x_testNARX__batches             = make_batches(x_test, batchSize)
    y_Realtime_training_batches     = make_batches(y_Realtime_training, batchSize)
    y_Realtime_testing_batches      = make_batches(y_Realtime_test, batchSize)
    y_Interval_training_batches     = make_batches(y_Interval_training, batchSize)
    y_Interval_testing_batches      = make_batches(y_Interval_test, batchSize)
    y_Interpolate_training_batches  = make_batches(y_Interval_Interpolated_training, batchSize)
    y_Interpolate_testing_batches   = make_batches(y_Interval_Interpolated_test, batchSize)
    x_input_NARX_batches            = make_batches(x_input_NARX, batchSize)
    
    
    
    return x_input_batches,x_test_batches,x_input_NARX_batches,y_Realtime_training_batches,y_Realtime_testing_batches,y_Interval_training_batches, y_Interval_testing_batches,y_Interpolate_training_batches, y_Interpolate_testing_batches


#%% Plot results
def plot_test_data(gas_sample, actual, predict):
    plt.title("Forecast vs Actual, gas " + str(gas_sample), fontsize=14)
    plt.plot(pd.Series(np.ravel(actual[:,gas_sample])), "bo", markersize=1, label="Actual")
    plt.plot(pd.Series(np.ravel(predict[:,gas_sample])), "r.", markersize=1, label="Forecast")
    plt.legend(loc="upper left")
    plt.xlabel("Time Periods")
    plt.show()


#%% TensorBoard summaries for a given variable
def variable_summaries(var):
    #tf.summary.scalar('value',var)
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)