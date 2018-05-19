"""
Author:         David Beam, db4ai
Date:           18 January 2018
Description:    Functions used to create, train, and test a multilayer perceptron neural network
"""

import datetime
import numpy as np
import tensorflow as tf

#%% Create arrays containing the weights and biases for the MLP
def create_weights_biases(numInputs, hidden_layer_widths, numOutputs, mean, std_dev):
    # Create and initialize the weights of the NN
    with tf.name_scope('weights'):
        weights = {}
        for i in range(len(hidden_layer_widths)+1):
            # Add hidden layers
            if i < len(hidden_layer_widths):
                # First layer, size by the number of inputs to the NN
                if i==0:
                    key = 'h' + str(i+1)
                    weights[key]  = tf.Variable(tf.random_normal([numInputs, hidden_layer_widths[i]],mean,std_dev))
                # Hidden layers
                else:
                    key = 'h' + str(i+1)
                    weights[key]  = tf.Variable(tf.random_normal([hidden_layer_widths[i-1], hidden_layer_widths[i]],mean,std_dev))
            # Add the output layer
            else:
                key = 'out'
                weights['out'] = tf.Variable(tf.random_normal([hidden_layer_widths[i-1], numOutputs],mean,std_dev))
            # Add the variable to TensorBoard
            with tf.name_scope(key):
                variable_summaries(weights[key])
    # Create and initialize the weights of the NN
    with tf.name_scope('biases'):
        biases = {}
        for i in range(len(hidden_layer_widths)+1):
            # Add hidden layers
            if i < len(hidden_layer_widths):
                # Hidden layers
                key = 'b' + str(i+1)
                biases[key]  = tf.Variable(0.0)
                #biases[key]  = tf.Variable(tf.random_normal([hidden_layer_widths[i]],mean,std_dev))
            # Add the output layer
            else:
                key = 'out'
                biases['out'] = tf.Variable(0.0)
                biases['out'] = tf.Variable(tf.random_normal([numOutputs],mean,std_dev))
            # Add the variable to TensorBoard
            with tf.name_scope(key):
                variable_summaries(biases[key])
    return weights,biases


#%% Execute a multilayer perceptron network for the given input, weights, biases and dropout keep probability
def multilayer_perceptron(x, weights, biases, keep_prob):
    with tf.name_scope('layers'):
        # Initialize the first layer to the input array
        layer = x
        # Loop through the layers and apply the weights and biases
        for i in range(len(weights)):
            # Create a layer name for TensorBoard
            layer_name = 'layer' + str(i+1)
            with tf.name_scope(layer_name):
                # Apply the weights and biases
                if i<len(weights)-1:
                    layer = tf.add(tf.matmul(layer, weights['h'+str(i+1)]), biases['b'+str(i+1)])
                else:
                    layer = tf.add(tf.matmul(layer, weights['out']), biases['out'])
    return layer


#%% Train MLP network
def multilayer_perceptron_train(sess,cfg,saver,summary_writer,x,y,keep_prob,x_batches,y_batches,optimizer,cost,merged_summary_op):    
    #  Loop through each epoch
    for epoch in range(cfg.epochs):
        avg_cost = 0.0
        for i in range(x_batches.shape[0]):
            batch_x, batch_y = x_batches[i,:,:], y_batches[i,:,:]
            _, c, summary = sess.run([optimizer, cost, merged_summary_op], 
                            feed_dict={
                                x: batch_x, 
                                y: batch_y, 
                                keep_prob: cfg.dropout_output_keep_prob
                            })
            avg_cost += c / x_batches.shape[0]
        if epoch % cfg.display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        # Save the model and save the TensorBoard summaries
        save_path = saver.save(sess, cfg.dir_path + cfg.model_dir + cfg.model_name)
        summary_writer.add_summary(summary, epoch)
        summary_writer.flush()

    
#%% Test the MLP network
def multiplayer_perceptron_test(sess,cfg,summary_writer,x,keep_prob,x_input,y_shape,predictions):
    # Run the predictions for a batch style input (3 dimensions [batch, timestep, sample])
    if len(x_input.shape)==3:
        predicted = np.zeros(shape=(y_shape[0],y_shape[1],y_shape[2]), dtype=float)
        for i in range(x_input.shape[0]):
            batch_testx = x_input[i,:,:]
            temp = sess.run([predictions], 
                            feed_dict={
                                x: batch_testx,
                                keep_prob: 1.0
                            })
            predicted[i,:,:] = temp[0]
        return np.reshape(predicted,(predicted.shape[0]*predicted.shape[1],y_shape[2]))
    # Run the predictions for a non-batch style input (2 dimensions [timestep, sample])
    elif len(x_input.shape)==2:
        predicted = np.zeros(shape=(y_shape[0],y_shape[1]), dtype=float)
        for i in range(x_input.shape[0]):
            row_x = x_input[i,:]
            row_x = np.reshape(np.append(row_x, row_x,axis=0),[-1,len(row_x)])
            temp = sess.run([predictions], 
                            feed_dict={
                                x: row_x,
                                keep_prob: 1.0
                            })
            predicted[i,:] = temp[0][0,:]
        return predicted
    # Unknown dimension, Error
    else:
        return 0.0

#%% TensorBoard summaries for a given variable
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)