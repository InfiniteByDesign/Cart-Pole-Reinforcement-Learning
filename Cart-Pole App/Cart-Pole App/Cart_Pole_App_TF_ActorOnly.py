#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:         David Beam, db4ai
Date:           18 January 2018
Description:    Create a MLP NN, train using the supplied dataset then test and display the test results on a chart
                Uses Keras and a simple cost function based off of angle
"""
# Include files
import gym
import numpy as np
from random import uniform

from keras.models import Sequential
from keras.layers import Dense


# Include custom files
import functions as func
import Configuration as cfg
import StateModel as sm

func.print_header()

# Create the CartPole environment for the physics model
if cfg.useOpenAImodel:
    env = gym.make('CartPole-v0')

print("Defining Model")
   
# create model
model = Sequential()
model.add(Dense(100, input_dim=cfg.action_input_size, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train and Test the model

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
    Y_hist      = []
    X_hist      = []
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
        act = model.predict(X)
        
        # Force is either +- 10 but the u is set to +-1 to normalize, action_u is the OpenAI input
        if act[0][0] >= 0.5:
            _u = 10
            u = 1.0
            action_u = 1
        elif act[0][0] < 0.5:
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
            #_X       = np.array([np.rad2deg(state[2]),np.rad2deg(state[3]),state[0],state[1]]).reshape(1,4)
        else:
            # Use the custom Cart-Pole model
            # state = [ang, ang vel, dist, vel]
            state   = np.array(sm.cart_pole_model(cfg.dt,last_state,0,_u)).reshape(1,5)
            X       = np.array([state[0,0],state[0,1],state[0,2],state[0,3]]).reshape(1,4)
            #_X       = np.array([state[0,0],state[0,1],state[0,2],state[0,3]]).reshape(1,4)
        
        # Determine the success feedback, r
        # state = [ang, ang vel, dist, vel]
        angle = X[0,0]                                        
        if angle < -12 or angle > 12 or X[0,2] < -2.4 or X[0,2] > 2.4:
            _r = -1
        else:
            _r = 0
            
        # Critic, create the critic input and evaluate the network
        _r = np.array(_r).reshape(1,1)
        J  = angle**2 + X[0,2]**2
        
        if J-jlast > 0:
            Y = -(act-0.5) + 0.5
        else:
            Y = act
                 
        
        if i % cfg.display_step == 0 and i>1:
            print("Epoch:", '%04d' % (epoch+1), "max was:", '%06d' % (max_i + 1), "steps, this epoch was:", '%06d' % (i + 1), "Ang:", "{:.3f}".format(angle), "Dist:", "{:.3f}".format(X[0,2]), "force was:", '%03d' % (_u), "CTG:", "{:.5f}".format(J))
        
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
        jlast       = J
        last_state  = state
        last_X      = X

        # Save variables for training
        X_hist.append(X)
        Y_hist.append(Y)
        
        # Increment the index
        i = i + 1
        
    # Train the model a set number of times on the current states
    X_hist = np.asarray(X_hist)
    X_hist.resize((X_hist.shape[0],4))
    Y_hist = np.asarray(Y_hist)
    Y_hist.resize((Y_hist.shape[0],1))
    model.fit(X_hist, Y_hist, nb_epoch=100, batch_size=len(X_hist),  verbose=0)
            
#tensorboard --logdir '/Users/David/Documents/GitHub/Cart-Pole-Reinforcement-Learning/Cart-Pole App/Cart-Pole App/logs'