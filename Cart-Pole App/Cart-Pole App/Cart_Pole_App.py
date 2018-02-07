
"""
Author:         David Beam, db4ai
Date:           18 January 2018
Description:    Create a MLP NN, train using the supplied dataset then test and display the test results on a chart
"""
# Include files
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import uniform

# Include custom files
import functions as func
import Configuration as cfg
import MLP_Definition as mlp
import StateModel as sm

func.print_header()

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
    with tf.name_scope('Model'):
        action_w, action_b = mlp.create_weights_biases(4, cfg.hidden_layer_widths, 1, cfg.init_weights_bias_mean_val, cfg.init_weights_bias_std_dev)
        critic_w, critic_b = mlp.create_weights_biases(6, cfg.hidden_layer_widths, 1, cfg.init_weights_bias_mean_val, cfg.init_weights_bias_std_dev)
        
    # -------------------------------------------------------------------------
    # Critic Network
    
    # The Cost-to-GO fuction (J)
    with tf.name_scope('Critic-Network'):
        cost_to_go = mlp.multilayer_perceptron(x_c, critic_w, critic_b, keep_prob) 
        func.variable_summaries(cost_to_go)
        
    # The Cost function of the Critic NN
    with tf.name_scope('Critic-Cost'):
        critic_cost = tf.reduce_sum(tf.square(cfg.alpha * cost_to_go - (Jlast-r)))
        func.variable_summaries(critic_cost)                           
    
    # The Critic Optimization algorithm
    with tf.name_scope('Critic-Optimizer'):
       critic_opt_op = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate).minimize(critic_cost) 
   
    # -------------------------------------------------------------------------
    # Action Network
    
    # The Action function (u)
    with tf.name_scope('Action-Network'):
        action = mlp.multilayer_perceptron(x_a, action_w, action_b, keep_prob) 
        func.variable_summaries(action)
        
    # The Cost function of the Action NN
    with tf.name_scope('Action-cost'):
        action_cost = tf.reduce_sum(tf.square(cost_to_go))
        func.variable_summaries(action_cost)
        
    # The Action Optimization algorithm
    with tf.name_scope('Action-Optimizer'):
        action_opt_op = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate).minimize(action_cost) 
    # -------------------------------------------------------------------------

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
        done = False
        max_i = 0
        for epoch in range(cfg.epochs):
            if done == True:
                break
            
            # Initial State
            t = 0
            dt = 0.02
            jlast = np.array([0]).reshape(1,1)
            state = np.array([uniform(-np.deg2rad(12),np.deg2rad(12)),0,0,0,0]).reshape(1,5) #[angle, ang_vel, dist, vel, ang_acc]
            simple_state = np.array([state[0,0],state[0,1],state[0,2],state[0,3]]).reshape(1,4)
            
            i = 0
            fail = False
            
            # Loop through the iterations until fail or pass
            while fail==False and i < 1000:
                # Action
                act = sess.run([action], feed_dict={x_a: simple_state, keep_prob: cfg.dropout_output_keep_prob})
                u = 0
                if act[0][0][0] > 0:
                    u = 10
                elif act[0][0][0] < 0:
                    u = -10            
                
                # Calculate the change in state then the new state matrix
                last_state  = simple_state
                state       = np.array(sm.cart_pole_model(dt,state,0,u)).reshape(1,5)
                simple_state = np.array([state[0,0],state[0,1],state[0,2],state[0,3]]).reshape(1,4)
                
                # Determine the success feedback, r
                # state = [ang, ang vel, dist, vel]
                if state[0,0] <= 12 and state[0,0] >= -12:
                    _r = 0
                else:
                    _r = -1
                    
                # Break the loop if we fail to keep the angle in range
                if _r == -1:
                    fail = True
                    if i > max_i:
                        max_i = i
                
                # Critic, create the critic input and evaluate the network
                act = np.array(act[0][0][0]).reshape(1,1)
                _r = np.array(_r).reshape(1,1)
                temp = np.concatenate((act,_r),axis=1)
                critic_input    = np.concatenate((simple_state,temp),axis=1)
                critic_output   = sess.run([cost_to_go], feed_dict={x_c: critic_input, keep_prob: cfg.dropout_output_keep_prob})
                critic_output   = np.array(critic_output[0]).reshape(1,1)
                
                # Update the weights and biases
                a_cost, summary = sess.run([action_opt_op, merged_summary_op], feed_dict={x_a: last_state, x_c: critic_input, Jlast: jlast, r: _r}) 
                c_cost, summary = sess.run([critic_opt_op, merged_summary_op], feed_dict={x_a: last_state, x_c: critic_input, Jlast: jlast, r: _r})          
                
                # Action weights update
                #for wc2 in range(cfg.hidden_layer_widths):
                #    action_w[1,wc2] = cfg.alpha * critic_error * action_w[1,wc2]
                #for wc1 in range(cfg.hidden_layer_widths):
                
                
                if i % cfg.display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "Iteration:", '%04d' % (i+1), "force=", '%03d' % (u),"Ang:", "{:.3f}".format(state[0,0]), "AngVel:", "{:.3f}".format(state[0,1]), "Dist=", "{:.3f}".format(state[0,2]), "Vel=", "{:.3f}".format(state[0,3]), "Action=", "{:.3f}".format(act[0][0]), "CTG=", "{:.3f}".format(critic_output[0][0]))
                if fail == True:
                    print("----------------------------------------------------")
                    print("Fail after:", '%04d' % (i+1), " steps, max was:", '%04d' % (max_i + 1))
                    print("----------------------------------------------------")
                    
                
                summary_writer.add_summary(summary, epoch)
                summary_writer.flush()
            
                # Increment the time index and save variables
                t = t + dt
                jlast = critic_output
        
                # Increment the index
                i = i + 1
            
            # Save the model and save the TensorBoard summaries
            save_path = saver.save(sess, cfg.dir_path + cfg.model_dir + cfg.action_model_name)
            save_path = saver.save(sess, cfg.dir_path + cfg.model_dir + cfg.critic_model_name)
                
    # Flushes the summaries to disk and closes the SummaryWriter
    summary_writer.close()