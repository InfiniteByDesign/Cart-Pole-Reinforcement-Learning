#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:         David Beam, db4ai
Date:           18 January 2018
Description:    Create a MLP NN, train using the supplied dataset then test and display the test results on a chart
"""
# Include files
import gym
import numpy as np
import tensorflow as tf
from random import uniform

# Include custom files
import functions as func
import Configuration as cfg
import MLP_Definition as mlp
import StateModel as sm

func.print_header()

# Create the CartPole environment for the physics model
if cfg.useOpenAImodel:
    env = gym.make('CartPole-v0')

# Explicitly create a Graph object
tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():

    print("Defining Variables")
       
    # Inputs, Placeholders for the input, output and drop probability
    with tf.name_scope('input'):
        x_a     = tf.placeholder(tf.float32, shape=[1, 4], name="input-state_action") 
        x_c     = tf.placeholder(tf.float32, shape=[1, 6], name="input-state_critic") 
        r       = tf.placeholder(tf.float32, shape=[1, 1], name="reinforcement")
        J       = tf.placeholder(tf.float32, shape=[1, 1], name="cost-to-go")
        Jlast   = tf.placeholder(tf.float32, shape=[1, 1], name="last-cost-to-go")
        
        # Dropout Keep Pobability
        keep_prob   = tf.placeholder("float")
        last_ctg    = tf.placeholder("float")

    print("Defining Model")
        
    # Setup the NN Model
    with tf.name_scope('Action-Model'):
        action_w, action_b = mlp.create_weights_biases(cfg.action_input_size, cfg.hidden_layer_widths, 1, cfg.init_weights_bias_mean_val, cfg.init_weights_bias_std_dev)
    
    with tf.name_scope('Critic-Model'):    
        critic_w, critic_b = mlp.create_weights_biases(cfg.critic_input_size, cfg.hidden_layer_widths, 1, cfg.init_weights_bias_mean_val, cfg.init_weights_bias_std_dev)
        
    # -------------------------------------------------------------------------
    # Critic Network
    
    # The Cost-to-GO fuction (J)
    with tf.name_scope('Critic-Output'):
        cost_to_go = mlp.multilayer_perceptron(x_c, critic_w, critic_b, keep_prob) 
        func.variable_summaries(cost_to_go)
        
    # The Cost function of the Critic NN
    with tf.name_scope('Critic-Cost'):
        critic_cost = tf.reduce_sum(0.5*tf.square(cfg.alpha * cost_to_go - (Jlast-r)))
        #critic_cost = cfg.alpha * cost_to_go - (Jlast-r)
        func.variable_summaries(critic_cost)                           
    
    # The Critic Optimization algorithm
    with tf.name_scope('Critic-Optimizer'):
       critic_opt_op = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate).minimize(critic_cost) 
   
    # -------------------------------------------------------------------------
    # Action Network
    
    # The Action function (u)
    with tf.name_scope('Action-Output'):
        action = mlp.multilayer_perceptron(x_a, action_w, action_b, keep_prob) 
        func.variable_summaries(action)
        
    # The Cost function of the Action NN
    with tf.name_scope('Action-Cost'):
        action_cost = tf.reduce_sum(0.5*tf.square(cost_to_go))
        #action_cost = tf.reduce_sum(-cost_to_go*x_a)
        #action_cost = -cost_to_go
        func.variable_summaries(action_cost)
        
    # The Action Optimization algorithm
    with tf.name_scope('Action-Optimizer'):
        action_opt_op = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate).minimize(action_cost) 
    # -------------------------------------------------------------------------

    # Tensorboard functions and model saving functions
    merged_summary_op = tf.summary.merge_all()                                           
    summary_writer = tf.summary.FileWriter(cfg.dir_path + cfg.log_dir, graph)  

    print("Starting TensorFlow Session")  

    # Train and Test the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        print("Training Model")
        
        # Initialize session variables
        Done = False
        max_i = 0
        
        # Loop through the set number of epochs for this NN
        for epoch in range(cfg.epochs):
            
            # If we reached the max number of steps (600,000) then we are done, stop the epochs
            if Done == True:
                break
            
            # Initial State
            jlast       = np.array([0]).reshape(1,1)
            state       = np.array([uniform(-np.deg2rad(10),np.deg2rad(10)),0,0,0,0]).reshape(1,5) #[angle, ang_vel, dist, vel, ang_acc]
            X           = np.array([state[0,0],state[0,1],state[0,2],state[0,3]]).reshape(1,4)
            last_state  = state
            last_X      = X
            
            i = 0
            fail = False
            
            # Reset the OpenAI environment
            if cfg.useOpenAImodel:
                observation = env.reset()
                env._max_episode_steps = 600001
            
            # Loop through the iterations until fail or pass
            while fail==False and i < 60000:
                # Train on each state several times to allow for multiple training attempts on a failure scenario
                                    
                # Action
                act = sess.run([action], feed_dict={x_a: last_X, keep_prob: cfg.dropout_output_keep_prob})
                
                # Force is either +- 10 but the u is set to +-1 to normalize, action_u is the OpenAI input
                if act[0][0][0] >= 0:
                    _u = 10
                    u = 1.0
                    action_u = 1
                elif act[0][0][0] < 0:
                    _u = -10  
                    u = -1.0
                    action_u = 0
                
                # Render the OpenAI movie
                if cfg.renderOpenAImodel:
                    env.render()
                    
                # Calculate the change in state then the new state matrix
                if cfg.useOpenAImodel:
                    # Use the OpenAI Cart-Pole model
                    # state = [x, xdot, theta, thedadot]
                    state, reward, done, info = env.step(action_u)
                    X       = np.array([np.rad2deg(state[2]),np.rad2deg(state[3]),state[0],state[1]]).reshape(1,4)
                else:
                    # Use the custom Cart-Pole model
                    # state = [ang, ang vel, dist, vel]
                    state   = np.array(sm.cart_pole_model(cfg.dt,last_state,0,_u)).reshape(1,5)
                    X       = np.array([state[0,0],state[0,1],state[0,2],state[0,3]]).reshape(1,4)
                
                # Determine the success feedback, r
                # state = [ang, ang vel, dist, vel]
                angle = X[0,0]                                        
                if angle < -12 or angle > 12 or X[0,2] < -2.4 or X[0,2] > 2.4:
                    _r = -1
                else:
                    _r = 0
                    
                # Critic, create the critic input and evaluate the network
                _r = np.array(_r).reshape(1,1)
                temp = np.concatenate((np.array(u).reshape(1,1),_r),axis=1)
                critic_input    = np.concatenate((X,temp),axis=1)
                critic_output   = sess.run([cost_to_go], feed_dict={x_c: critic_input, keep_prob: cfg.dropout_output_keep_prob})
                critic_output   = np.array(critic_output[0]).reshape(1,1)
                    
                # Train the model a set number of times on the current states
                for update in range(cfg.retrain_state):
                    # Update the weights and biases
                    a_cost, summary = sess.run([action_opt_op, merged_summary_op], feed_dict={x_a: X, x_c: critic_input, Jlast: jlast, r: _r}) 
                    c_cost, summary = sess.run([critic_opt_op, merged_summary_op], feed_dict={x_a: X, x_c: critic_input, Jlast: jlast, r: _r})  
                    
                # Write the data to tensorboard log files
                summary_writer.add_summary(summary, epoch)
                summary_writer.flush()        
                
                if i % cfg.display_step == 0 and i>1:
                    print("Epoch:", '%04d' % (epoch+1), "max was:", '%06d' % (max_i + 1), "steps, this epoch was:", '%06d' % (i + 1), "Ang:", "{:.3f}".format(angle), "Dist:", "{:.3f}".format(X[0,2]), "force was:", '%03d' % (_u), "CTG:", "{:.5f}".format(critic_output[0,0]))
                
                # If negative feedback then the trial failed
                if _r == -1:
                    fail = True
                    
                    # Print a summary
                    print("Epoch:", '%04d' % (epoch+1), "max was:", '%06d' % (max_i + 1), "steps, this epoch was:", '%06d' % (i + 1))

                    # Save the number of steps, i, if this is the max so far
                    if i > max_i:
                        max_i = i  
                
                # Check if we reached the max time step
                if i == 600000:
                    done = True
                    print("Epoch:", '%04d' % (epoch+1), " MAX STEP COUNT REACHED, 600,000!")
            
                # Save variables to reuse
                jlast       = critic_output
                last_state  = state
                last_X      = X
        
                # Increment the index
                i = i + 1
                
    # Flushes the summaries to disk and closes the SummaryWriter
    summary_writer.close()
    #tensorboard --logdir '/Users/David/Documents/GitHub/Cart-Pole-Reinforcement-Learning/Cart-Pole App/Cart-Pole App/logs'