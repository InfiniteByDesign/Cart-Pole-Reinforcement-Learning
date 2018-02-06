"""
Author:         David Beam, db4ai
Date:           18 January 2018
Description:    Create a MLP NN, train using the supplied dataset then test and display the test results on a chart
"""
# Include files
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Include custom files
import CSVReader as csvreader
import functions as func
import Configuration as cfg
import MLP_Definition as mlp

func.print_header()

# Explicitly create a Graph object
tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    
    print("Loading Data")
        
    #%% Import the data, specific to the dataset being used for the NN.
    #   Modify this code and the CSVReader to your specific dataset
    x1,x2,x2_Interval,y_Realtime,y_Interval,y_Interval_interpolated,numInputs,numOutputs,trainingSamples,testingSamples = \
        csvreader.Import_CSV(cfg.dir_path, cfg.dir_char,cfg.filename,cfg.trainingPct)
    
    #%% Pre-process the data    
    x_training_batchs,x_test_batches,                              \
    x_training_NARX_batches,                                       \
    y_Realtime_training_batches,y_Realtime_testing_batches,        \
    y_Interval_training_batches,y_Interval_testing_batches,        \
    y_Interpolate_training_batches,y_Interpolate_testing_batches = \
        func.split_data_into_batches(x1,y_Realtime,y_Interval,y_Interval_interpolated,x2,x2_Interval,trainingSamples,testingSamples,cfg.batchSize,cfg.numInputDelays,cfg.numOutputDelays)

    # Choose from the datasets above for the x and y data
    x_train     = x_training_batchs
    x_test      = x_test_batches
    #y_train     = y_Realtime_training_batches
    #y_test      = y_Realtime_testing_batches
    #y_train     = y_Interval_training_batches
    #y_test      = y_Interval_testing_batches
    y_train     = y_Interpolate_training_batches
    y_test      = y_Interpolate_testing_batches

    # Determine the number of samples for testing and training
    trainingSamples = int(len(x1) * cfg.trainingPct / 100)
    testingSamples  = len(x1) - trainingSamples
       
    print("Defining Variables")
       
    # Inputs, Placeholders for the input, output and drop probability
    with tf.name_scope('input'):
        # Input, size determined by batch size and number of inputs per time step
        x = tf.placeholder(tf.float32, shape=[None, numInputs], name="x-input") 
        # Output, size determined by batch size and number of outputs per time step
        y = tf.placeholder(tf.float32, shape=[None, numOutputs], name="y-input")
        # Dropout Keep Pobability
        keep_prob = tf.placeholder("float")

    print("Defining Model")
        
    # Setup the NN Model
    with tf.name_scope('Model'):
        weights, biases = mlp.create_weights_biases(numInputs, cfg.hidden_layer_widths, numOutputs, cfg.init_weights_bias_mean_val, cfg.init_weights_bias_std_dev)
        
    # The Prediction function
    with tf.name_scope('predictions'):
        predictions = mlp.multilayer_perceptron(x, weights, biases, keep_prob) 
        func.variable_summaries(predictions)

    # The Cost fuction
    with tf.name_scope('cost'):
        cost = tf.reduce_sum(tf.square(predictions - y))
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
        func.variable_summaries(cost)
    
    # The Optimization algorithm
    with tf.name_scope('optimzer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate).minimize(cost)                            

    # Tensorboard functions and model saving functions
    merged_summary_op = tf.summary.merge_all()                                           
    summary_writer = tf.summary.FileWriter(cfg.dir_path + cfg.log_dir, graph)  
    saver = tf.train.Saver(max_to_keep=1)      

    print("Starting TensorFlow Session")  

    # Train and Test the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        print("Training Model")
        
        # Train the Model
        mlp.multilayer_perceptron_train(sess,cfg,saver,summary_writer,x,y,keep_prob,x_train,y_train,optimizer,cost,merged_summary_op)
        
        print("Optimization Finished")
        print("Running Test Data...")

        # Test the Model
        pred = mlp.multiplayer_perceptron_test(sess,cfg,summary_writer,x,keep_prob,x_test,y_test.shape,predictions)
        
        # Display the results
        actual = np.reshape(y_Interval_testing_batches,(y_test.shape[0]*y_test.shape[1],y_test.shape[2]))
        interp = np.reshape(y_Interpolate_testing_batches,(y_test.shape[0]*y_test.shape[1],y_test.shape[2]))
        func.plot_test_data(0,actual,pred)
        func.plot_test_data(0,interp,pred)

    # Flushes the summaries to disk and closes the SummaryWriter
    summary_writer.close()