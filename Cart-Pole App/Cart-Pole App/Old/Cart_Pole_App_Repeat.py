
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


def action_update(action_w1, action_w2, critic_factor, error, X, u, g, alpha, width): 
    # Change in w2
    d_w2 = (0.5 * error * (1 - np.power(u,2)) * g * critic_factor).reshape(width,1)
    # Change in w1
    d_w1 = np.outer(X,0.5 * error * (1 - np.power(u,2)) * action_w2 * 0.5 * (1 - np.power(g.reshape(width,1),2)) * critic_factor)
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

def critic_update(critic_w1, critic_w2, error, x_a, _p, alpha, width):
    # Change in w2
    d_w2    = (alpha*error * _p).reshape(width,1)
    # Change in w1
    temp_a  = x_a.reshape(6,1)
    temp_b  = alpha * error * action_w2 * (0.5*(1-np.power(_p,2).reshape(width,1)))
    d_w1    = np.outer(temp_a,temp_b)
    # Normalize the weights
    w1 = (critic_w1 + d_w1) / np.linalg.norm(critic_w1 + d_w1, ord=1)
    w2 = (critic_w2 + d_w2) / np.linalg.norm(critic_w2 + d_w2, ord=1)
    # Compute the critic factor used to update the action network
    critic_factor = np.sum( 0.5*w2*(1-np.power(_p,2)) * w1[4,:] )
    # Output
    return w1, w2, critic_factor
  
# -------------------------------------------------------------------------
# Setup Model Functions
    
# Setup the NN Model, Uniform Distribution
def define_weights_uniform(minv,maxv,action_in,action_width,critic_in,critic_width):
    action_w1 = np.ones((action_in,action_width),dtype=float) * np.random.uniform(minv,maxv,(action_in,action_width))
    action_w2 = np.ones((action_width,1),dtype=float) * np.random.uniform(minv,maxv,(action_width,1))
    critic_w1 = np.ones((critic_in,critic_width),dtype=float) * np.random.uniform(minv,maxv,(critic_in,critic_width))
    critic_w2 = np.ones((critic_width,1),dtype=float) * np.random.uniform(minv,maxv,(critic_width,1))
    return action_w1, action_w2, critic_w1, critic_w2

# Setup the NN Model, Normal Distribution
def define_weights_normal(mean,variance,action_in,action_width,critic_in,critic_width):
    action_w1 = np.ones((action_in,action_width),dtype=float) * np.random.normal(mean,variance,(action_in,action_width))
    action_w2 = np.ones((action_width,1),dtype=float) * np.random.normal(mean,variance,(action_width,1))
    critic_w1 = np.ones((critic_in,critic_width),dtype=float) * np.random.normal(mean,variance,(critic_in,critic_width))
    critic_w2 = np.ones((critic_width,1),dtype=float) * np.random.normal(mean,variance,(critic_width,1))
    return action_w1, action_w2, critic_w1, critic_w2


# -------------------------------------------------------------------------
# Run the program

func.print_header()

# Explicitly create a Graph object
tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():

    print("Defining Parameters")
    
    # NN Learning Parameters
    alpha           = 0.99
    learning        = 0.01
    # NN sizes parameters
    action_in       = 4
    action_width    = 4
    critic_in       = 6
    critic_width    = 4
    # Min and max values for weights Uniform Distribution
    minv = -0.1
    maxv =  0.1
    # Mean and variance for weights Normal Distribution
    mean = 0.0
    variance = 0.5
    # Number of times to evaulate each state and update the weights
    run_state = 25

    # Placeholders
    best_angle_hist = []
    best_vel_hist = []
    best_j_hist = []
    done = False
    max_i = 0        

    print("Defining Model") 
    
    # Selection to start with only the weights from the best run
    action_w1, action_w2, critic_w1, critic_w2 = define_weights_uniform(minv,maxv,action_in,action_width,critic_in,critic_width)        
        
    print("Training Model")
    
    for epoch in range(cfg.epochs):
        if done == True:
            break
        
        # Initial States
        i = 0
        fail = False
        t = 0
        dt = 0.02
        Jlast = np.array([0]).reshape(1,1)
        state = np.array([uniform(-np.deg2rad(5),np.deg2rad(5)),0,0,0,0]).reshape(1,5) #[angle, ang_vel, dist, vel, ang_acc]
        X = np.array([state[0,0],state[0,1],state[0,2],state[0,3]]).reshape(1,4)
        laststate = state
        
        # Placeholders
        angle_hist = []
        vel_hist = []
        j_hist = []
        
        # Loop through the iterations until fail or pass
        while fail==False and i < 600000:
            # Run the current state a set number of times before moving to the next state
            # This allows for more back-prop of the weights for bad (r=-1) cases
            for update in range(run_state):
                # Action
                u, g = action_output(action_w1,action_w2,X)
                if u >= 0:
                    _u = 10
                elif u < 0:
                    _u = -10   
                    
                # Calculate the change in state then the new state matrix
                last_X  = X
                state   = np.array(sm.cart_pole_model(dt,laststate,0,_u)).reshape(1,5)
                X       = np.array([state[0,0],state[0,1],state[0,2],state[0,3]]).reshape(1,4)
                
                # Determine the success feedback, r
                # state = [ang, ang vel, dist, vel, ang_acc]
                angle = np.rad2deg(state[0,0])%360
                if angle > 180:
                    angle = angle - 360
                if angle <= 12 and angle >= -12 and state[0,2]>-2.4 and state[0,2]<2.4:
                    r = 0
                else:
                    r = -1
                
                # Critic, create the critic input and evaluate the network
                temp            = np.array([u,r],dtype=float)
                critic_input    = np.concatenate((X,temp.reshape((1,2))),axis=1)
                J, _p           = critic_output(critic_w1,critic_w2,critic_input)
                
                # Calculate the error
                #Ea = -J
                Ea = action_cost(J)
                #Ec = -(alpha*J - (Jlast-r))
                Ec = critic_cost(alpha, J, Jlast, r)
                
                # Update the weights
                critic_w1, critic_w2, critic_factor = critic_update(critic_w1, critic_w2, Ec, critic_input, _p, learning, critic_width)
                action_w1, action_w2 = action_update(action_w1, action_w2, critic_factor, Ea, X, u, g, learning, action_width)
               
            # Save history
            angle_hist.append(angle)
            vel_hist.append(state[0,3])
            j_hist.append(J[0,0])

            # Break the loop if we fail to keep the angle in range
            if r == -1:
                fail = True
                # Print a summary
                print("Epoch:", '%04d' % (epoch+1), "max was:", '%06d' % (max_i + 1), "steps, this epoch was:", '%06d' % (i + 1))
                #print("----------------------------------------------------")
                #print("Epoch:", '%04d' % (i+1), " steps, max was:", '%04d' % (max_i + 1))
                #print("----------------------------------------------------")
                # Save best run only
                if i > max_i:
                    max_i = i
                    best_angle_hist = angle_hist
                    best_vel_hist = vel_hist
                    best_j_hist = j_hist
            
            # Check if we reached the max time step
            if i == 600000:
                done = True
                print("Epoch:", '%04d' % (epoch+1), " MAX STEP COUNT REACHED, 600,000!")
        
            # Increment the time index and save variables
            i = i + 1
            t = t + dt
            Jlast = J
            laststate = state
            
    plt.title("Best run, pendulum angle over time", fontsize=14)
    plt.plot(best_angle_hist)
    plt.show()
    plt.title("Best run, cart vel over time", fontsize=14)
    plt.plot(best_vel_hist)
    plt.show()
    plt.title("Best run, cost-to-go", fontsize=14)
    plt.plot(best_j_hist)
    plt.show()
    
    