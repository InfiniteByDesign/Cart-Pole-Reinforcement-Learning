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
import matplotlib.pyplot as plt
from random import uniform

# Include custom files
import functions as func
import Configuration as cfg
import StateModel as sm

# -------------------------------------------------------------------------
# Action Network Functions
    
def action_output(w1, w2, X):
    hidden  = np.dot(X,w1)
    g       = (1-np.exp(-hidden))/(1+np.exp(-hidden))
    v       = np.dot(g,w2)
    u       = (1-np.exp(-v))/(1+np.exp(-v))
    return u, g

def action_cost(J):
    return 0.5 * J**2


def action_update(action_w1, action_w2, critic_factor, error, X, u, g): 
    # Change in w2
    d_w2 = (0.5 * error * (1 - np.power(u,2)) * g * critic_factor).reshape(24,1)
    # Change in w1
    d_w1 = np.outer(X,0.5 * error * (1 - np.power(u,2)) * action_w2 * 0.5 * (1 - np.power(g.reshape(24,1),2)) * critic_factor)
    # Normalize the weights
    w1 = (action_w1 + d_w1) / np.linalg.norm(action_w1 + d_w1, ord=1)
    w2 = (action_w2 + d_w2) / np.linalg.norm(action_w2 + d_w2, ord=1)
    
    return w1, w2


# -------------------------------------------------------------------------
# Other Functions
# Plot the best results
def plot_results(angle_hist,vel_hist,j_hist,u_hist,x_hist,aw1_hist,aw2_hist):
    plt.title("pendulum angle over time", fontsize=14)
    plt.plot(angle_hist)
    plt.show()
    """
    plt.title("cart vel over time", fontsize=14)
    plt.plot(vel_hist)
    plt.show()
    """
    plt.title("cost-to-go", fontsize=14)
    plt.plot(j_hist)
    plt.show()
    """
    plt.title("force over time", fontsize=14)
    plt.plot(u_hist)
    plt.show()
    plt.title("x-dist over time", fontsize=14)
    plt.plot(x_hist)
    plt.show()
    plt.title("action w1 mean over time", fontsize=14)
    plt.plot(aw1_hist)
    plt.show()
    plt.title("action w2 mean over time", fontsize=14)
    plt.plot(aw2_hist)
    plt.show()
    """
    
func.print_header()

print("Defining Variables")

# Initialize variables
alpha       = 0.9

print("Defining Model")
    
# Setup the NN Model
action_w1 = np.ones((4,24),dtype=float) * np.random.normal(cfg.init_weights_bias_mean_val,cfg.init_weights_bias_std_dev,(4,24))
action_w2 = np.ones((24,1),dtype=float) * np.random.normal(cfg.init_weights_bias_mean_val,cfg.init_weights_bias_std_dev,(24,1))
  
print("Training Model")

# Train the Model
best_angle_hist = []
best_vel_hist   = []
best_j_hist     = []
best_u_hist     = []
best_x_hist     = []
best_aw1_hist   = []
best_aw2_hist   = []
Done = False
max_i = 0
for epoch in range(cfg.epochs):
    if Done == True:
        break
    
    # Initial Values
    i = 0
    fail = False
    t = 0
    dt = 0.02
    Jlast = np.array([0]).reshape(1,1)
                   
    # Initial state of the physics model
    #state = np.array([uniform(-np.deg2rad(5),np.deg2rad(5)),0,0,0,0]).reshape(1,5) #[angle, ang_vel, dist, vel, ang_acc]
    state = np.array([0,0,0,0,0]).reshape(1,5) # angle,ang_vel,ang_acc,x,x_vel
    

    # Random initial force
    if np.random.uniform(0,1) < 0.5:
        u = 10
    else:
        u = -10
        
    # Calculate the change in state then the new state matrix
    state   = np.array(sm.cart_pole_model(dt,state[0,0],state[0,1],state[0,2],state[0,3],state[0,4],u)).reshape(1,5)
    X       = np.array([np.deg2rad(state[0,0]),state[0,1],state[0,3],state[0,4]]).reshape(1,4)
        
    # Placeholders
    angle_hist  = []
    vel_hist    = []
    j_hist      = []
    u_hist      = []
    x_hist      = []
    aw1_hist    = []
    aw2_hist    = []
    
    # Loop through the iterations until fail or pass
    while fail==False and i < 600000:
                    
        # Action
        u, g = action_output(action_w1,action_w2,X)
        if u >= 0:
            u = 10         # force
        elif u < 0:
            u = -10        # force
                    
        # Calculate the change in state then the new state matrix
        # state = [ang, ang vel, dist, vel]
        #state   = np.array(sm.cart_pole_model(dt,state[0,0],state[0,1],state[0,2],state[0,3],state[0,4],u)).reshape(1,5)
        state   = np.array(sm.cart_pole_model(dt,state[0,0],state[0,1],state[0,2],state[0,3],0,u)).reshape(1,5)
        X       = np.array([np.deg2rad(state[0,0]),state[0,1],state[0,3],state[0,4]]).reshape(1,4)
        
        # Determine the success feedback, r
        # state = [ang, ang vel, dist, vel, ang_acc]
        #angle = np.rad2deg(X[0,0])%360
        angle = X[0,0]%360
        if angle > 180:
            angle = angle - 360
        if angle <= 12 and angle >= -12 and X[0,2]>-2.4 and X[0,2]<2.4:
            r = 0
            update_range = 1
        else:
            r = -1
            update_range = 100
        
        # Critic, create the critic input and evaluate the network
        critic_input    = np.concatenate((X,np.array([u,r],dtype=float).reshape((1,2))),axis=1)
        J = angle**2 + X[0,2]**2
        
        # Calculate the action and critic error
        Ea = action_cost(J)
        
        # Update the weights
        
        for update in range(update_range):
            action_w1, action_w2 = action_update(action_w1, action_w2, 0.5, Ea, X, u, g)
       
        # Save history
        angle_hist.append(angle)
        vel_hist.append(X[0,3])
        j_hist.append(J)
        u_hist.append(u)
        x_hist.append(X[0,2])
        aw1_hist.append(np.mean(action_w1))
        aw2_hist.append(np.mean(action_w2))

        # Break the loop if we fail to keep the angle in range
        if r == -1:
            fail = True
            
            # Print a summary
            print("Epoch:", '%04d' % (epoch+1), "max was:", '%06d' % (max_i + 1), "steps, this epoch was:", '%06d' % (i + 1))

            # Save best run only
            if i > max_i:
                max_i = i
                best_angle_hist = angle_hist
                best_vel_hist   = vel_hist
                best_j_hist     = j_hist  
                best_u_hist     = u_hist
                best_x_hist     = x_hist
                best_aw1_hist   = aw1_hist
                best_aw2_hist   = aw2_hist
        
        """
        if i%100 == 0:
            plot_results(angle_hist,vel_hist,j_hist,u_hist,x_hist,aw1_hist,aw2_hist)
            temp = 1
        """
        
        # Check if we reached the max time step
        if i == 600000:
            Done = True
            print("Epoch:", '%04d' % (epoch+1), " MAX STEP COUNT REACHED, 600,000!")
    
        # Increment the time index and save variables
        i = i + 1
        t = t + dt
        Jlast = J
    
    # Done with one trial, loop back
    plot_results(angle_hist,vel_hist,j_hist,u_hist,x_hist,aw1_hist,aw2_hist)
    temp = 1
plot_results(best_angle_hist,best_vel_hist,best_j_hist,best_u_hist,best_x_hist,best_aw1_hist,best_aw2_hist)


