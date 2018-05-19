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
# Critic Network Functions
    
def critic_output(w1, w2, input):
    q = np.dot(input,w1)
    _p = (1-np.exp(-q))/(1+np.exp(-q))
    J = np.dot(_p,w2)
    return J, _p

def critic_cost(alpha, J, Jlast, r):
    return 0.5*(alpha*J - (Jlast-r))**2

def critic_update(critic_w1, critic_w2, error, x_a, _p, alpha):
    # Change in w2
    d_w2    = (alpha*error * _p).reshape(24,1)
    # Change in w1
    temp_a  = x_a.reshape(6,1)
    temp_b  = alpha * error * action_w2 * (0.5*(1-np.power(_p,2).reshape(24,1)))
    d_w1    = np.outer(temp_a,temp_b)
    # Normalize the weights
    w1 = (critic_w1 + d_w1) / np.linalg.norm(critic_w1 + d_w1, ord=1)
    w2 = (critic_w2 + d_w2) / np.linalg.norm(critic_w2 + d_w2, ord=1)
    # Compute the critic factor used to update the action network
    critic_factor = np.sum( 0.5*w2*(1-np.power(_p,2)) * w1[4,:] )
    # Output
    return w1, w2, critic_factor
    
# -------------------------------------------------------------------------
# Other Functions
# Plot the best results
def plot_results(angle_hist,vel_hist,j_hist,u_hist,x_hist,aw1_hist,aw2_hist,cw1_hist,cw2_hist):
    plt.title("pendulum angle over time", fontsize=14)
    plt.plot(angle_hist)
    plt.show()
    plt.title("cart vel over time", fontsize=14)
    plt.plot(vel_hist)
    plt.show()
    plt.title("cost-to-go", fontsize=14)
    plt.plot(j_hist)
    plt.show()
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
    plt.title("critic w1 mean over time", fontsize=14)
    plt.plot(cw1_hist)
    plt.show()
    plt.title("critic w2 mean over time", fontsize=14)
    plt.plot(cw2_hist)
    plt.show()
    
func.print_header()

print("Defining Variables")

# Initialize variables
alpha       = 0.9

print("Defining Model")
    
# Setup the NN Model
action_w1 = np.ones((4,24),dtype=float) * np.random.normal(cfg.init_weights_bias_mean_val,cfg.init_weights_bias_std_dev,(4,24))
action_w2 = np.ones((24,1),dtype=float) * np.random.normal(cfg.init_weights_bias_mean_val,cfg.init_weights_bias_std_dev,(24,1))
critic_w1 = np.ones((6,24),dtype=float) * np.random.normal(cfg.init_weights_bias_mean_val,cfg.init_weights_bias_std_dev,(6,24))
critic_w2 = np.ones((24,1),dtype=float) * np.random.normal(cfg.init_weights_bias_mean_val,cfg.init_weights_bias_std_dev,(24,1))
    
# Create the CartPole environment for the physics model
env = gym.make('CartPole-v0')
    
print("Training Model")

# Train the Model
best_angle_hist = []
best_vel_hist   = []
best_j_hist     = []
best_u_hist     = []
best_x_hist     = []
best_aw1_hist   = []
best_aw2_hist   = []
best_cw1_hist   = []
best_cw2_hist   = []
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
    state = np.array([0,0,0,0,0]).reshape(1,5) # angle,ang_vel,ang_acc,x,x_vel


    # Random initial force
    if np.random.uniform(0,1) < 0.5:
        u = 10
        action_u = 1
    else:
        u = -10
        action_u = 0
        
    # Calculate the change in state then the new state matrix
    # Use the OpenAI Cart-Pole model
    # state = [x, xdot, theta, thedadot]        
    observation = env.reset()
    env._max_episode_steps = 600001
    state, reward, done, info = env.step(action_u)
    X = np.array([np.rad2deg(state[2]),np.rad2deg(state[3]),state[0],state[1]]).reshape(1,4)

    # Placeholders
    angle_hist  = []
    vel_hist    = []
    j_hist      = []
    u_hist      = []
    x_hist      = []
    aw1_hist    = []
    aw2_hist    = []
    cw1_hist    = []
    cw2_hist    = []
    
    # Loop through the iterations until fail or pass
    while fail==False and i < 600000:
                    
        # Action
        u, g = action_output(action_w1,action_w2,X)
        if u >= 0:
            u = 10         # force
            action_u = 1    # OpenAI action state
        elif u < 0:
            u = -10        # force
            action_u = 0    # OpenAI action state
        
        # Render the OpenAI movie
        if cfg.renderOpenAImodel:
            env.render()
        
        # Calculate the change in state then the new state matrix
        # Use the OpenAI Cart-Pole model
        # state = [x, xdot, theta, thedadot]
        state, reward, done, info = env.step(action_u)
        X       = np.array([np.rad2deg(state[2]),np.rad2deg(state[3]),state[0],state[1]]).reshape(1,4)

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
        J, _p           = critic_output(critic_w1,critic_w2,critic_input)
        
        # Calculate the action and critic error
        Ea = action_cost(J)
        Ec = critic_cost(alpha, J, Jlast, r)
        
        # Update the weights
        for update in range(update_range):
            critic_w1, critic_w2, critic_factor = critic_update(critic_w1, critic_w2, Ec, critic_input, _p, 0.001)
            action_w1, action_w2 = action_update(action_w1, action_w2, critic_factor, 0.1*Ea, X, u, g)
       
        # Save history
        angle_hist.append(angle)
        vel_hist.append(X[0,3])
        j_hist.append(J[0,0])
        u_hist.append(u)
        x_hist.append(X[0,2])
        aw1_hist.append(np.mean(action_w1))
        aw2_hist.append(np.mean(action_w2))
        cw1_hist.append(np.mean(critic_w1))
        cw2_hist.append(np.mean(critic_w2))

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
                best_cw1_hist   = cw1_hist
                best_cw2_hist   = cw2_hist
        
        # Check if we reached the max time step
        if i == 600000:
            Done = True
            print("Epoch:", '%04d' % (epoch+1), " MAX STEP COUNT REACHED, 600,000!")
    
        # Increment the time index and save variables
        i = i + 1
        t = t + dt
        Jlast = J
    
    # Done with one trial, loop back
    #plot_results(angle_hist,vel_hist,j_hist,u_hist,x_hist,aw1_hist,aw2_hist,cw1_hist,cw2_hist)
    #temp = 1

plot_results(best_angle_hist,best_vel_hist,best_j_hist,best_u_hist,best_x_hist,best_aw1_hist,best_aw2_hist,best_cw1_hist,best_cw2_hist)
