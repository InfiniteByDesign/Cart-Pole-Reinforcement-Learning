"""
Author:         David Beam, db4ai
Date:           8 March 2018
Description:    Actor-Critic network functions for updating the networks,
                calculating the output and cost
"""

# Include files
import numpy as np
import gym

# -------------------------------------------------------------------------
# Actor Network Functions
# -------------------------------------------------------------------------
   
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
# ------------------------------------------------------------------------- 

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
# Actor-Critic Class
# ------------------------------------------------------------------------- 

class ActorCriticClass:
    " Class to hold all the objects and functions to train and use the Actor-Critic network"

    # -------------------------------------------------------------------------
    # Initialization Functions
    # -------------------------------------------------------------------------

    # Constructor
    def __init__(self):
        # Hyperparameters
        self.actor_width        = 0
        self.critic_width       = 0
        self.bias               = 0
        self.std_dev            = 0
        self.epochs             = 0
        self.retrain_state      = 0
        self.learning_rate      = 0        
        alpha                   = 0
        self.alpha              = 0
        self.max_episode_steps  = 0
        # Model Weights
        self.actor_w1           = []
        self.actor_w2           = []
        self.critic_w1          = []
        self.critic_w2          = []
        # History
        self.max_index          = 0
        self.best_angle_hist    = []
        self.best_vel_hist      = []
        self.best_j_hist        = []
        self.best_u_hist        = []
        self.best_x_hist        = []
        self.best_aw1_hist      = []
        self.best_aw2_hist      = []
        self.best_cw1_hist      = []
        self.best_cw2_hist      = []
        # Training Flow
        self.trainingDone       = False
        self.epoch              = 0
        self.env                = gym.make('CartPole-v0')
        self.renderOpenAImodel  = True

    # Initialize Weights, there are 4 inputs to the actor and 6 inputs to the critic
    def InitializeWeights(self):
        self.actor_w1  = np.ones((4,self.actor_width),dtype=float)  * np.random.normal(self.bias,self.std_dev,(4,self.actor_width))
        self.actor_w2  = np.ones((self.actor_width,1),dtype=float)  * np.random.normal(self.bias,self.std_dev,(self.actor_width,1))
        self.critic_w1 = np.ones((6,self.critic_width),dtype=float) * np.random.normal(self.bias,self.std_dev,(6,self.critic_width))
        self.critic_w2 = np.ones((self.critic_width,1),dtype=float) * np.random.normal(self.bias,self.std_dev,(self.critic_width,1))

    # Initialize the history values
    def InitializeHistory(self):
        self.max_index       = 0
        self.best_angle_hist = []
        self.best_vel_hist   = []
        self.best_j_hist     = []
        self.best_u_hist     = []
        self.best_x_hist     = []
        self.best_aw1_hist   = []
        self.best_aw2_hist   = []
        self.best_cw1_hist   = []
        self.best_cw2_hist   = []

    # -------------------------------------------------------------------------
    # Set Hyperparameters
    # -------------------------------------------------------------------------
    def Set_Hyperparameters(self):
        # General NN parameters
        self.epochs             = 100           # Number of iterations or training cycles, includes both the FeedFoward and Backpropogation
        self.actor_width        = 100
        self.critic_width       = 100
        self.bias               = 0.0           # Mean value of the normal distribution used to initialize the weights and biases
        self.std_dev            = 1.0           # Standard Deviation of the normal distribution used to initialize the weights and biases
        self.learning_rate      = 0.0001        # Learning Rate 
        self.alpha              = 0.0001        # cost-to-go discount factor
        self.max_episode_steps  = 600000
        self.retrain_state      = 10            # The number of times to repeat the training cycle for each physical state update

        # Cart Pole Physics Model parameters
        self.renderOpenAImodel = True       # Render a movie of the OpenAI progress, does not work in spyder gui

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    # Initialize values before a new training sequence
    def Train_Init(self):
        self.trainingDone   = False
        self.max_index      = 0
        self.epoch          = 0
                   
        # Initial state of the physics model
        self.state = np.array([0,0,0,0,0]).reshape(1,5) # angle,ang_vel,ang_acc,x,x_vel

        # Random initial force
        if np.random.uniform(0,1) < 0.5:
            action_u = 1
        else:
            action_u = 0
        
        # Calculate the change in state then the new state matrix
        # Use the OpenAI Cart-Pole model
        # state = [x, xdot, theta, thedadot]        
        self.observation = self.env.reset()
        self.env._max_episode_steps = self.max_episode_steps
        state, reward, done, info = self.env.step(action_u)
        self.X = np.array([np.rad2deg(state[2]),np.rad2deg(state[3]),state[0],state[1]]).reshape(1,4)

    # Train for one step
    def Train_Step(self):
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

        # Continue training if we are not done
        if self.epoch < self.epochs and self.trainingDone != True:
            # initialize values for this training step
            i = 0
            fail = False
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
            observation = self.env.reset()
            self.env._max_episode_steps = self.max_episode_steps+1
            state, reward, done, info = self.env.step(action_u)
            X = np.array([np.rad2deg(state[2]),np.rad2deg(state[3]),state[0],state[1]]).reshape(1,4)
    
            # Loop through the iterations until fail or pass
            while fail==False and i < self.max_episode_steps:
                    
                # Action
                u, g = action_output(self.actor_w1, self.actor_w2, X)
                if u >= 0:
                    u = 10          # force
                    action_u = 1    # OpenAI action state
                elif u < 0:
                    u = -10         # force
                    action_u = 0    # OpenAI action state
        
                # Render the OpenAI movie
                if self.renderOpenAImodel:
                    self.env.render()
        
                # Calculate the change in state then the new state matrix
                # Use the OpenAI Cart-Pole model
                # state = [x, xdot, theta, thedadot]
                state, reward, done, info = self.env.step(action_u)
                X = np.array([np.rad2deg(state[2]),np.rad2deg(state[3]),state[0],state[1]]).reshape(1,4)

                # Determine the success feedback, r
                # state = [ang, ang vel, dist, vel, ang_acc]
                # angle = np.rad2deg(X[0,0])%360
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
                J, _p           = critic_output(self.critic_w1,self.critic_w2,critic_input)
        
                # Calculate the action and critic error
                Ea = action_cost(J)
                Ec = critic_cost(self.alpha, J, Jlast, r)
        
                # Update the weights
                for update in range(update_range):
                    self.critic_w1, self.critic_w2, critic_factor = critic_update(self.critic_w1, self.critic_w2, Ec, critic_input, _p, self.alpha)
                    self.actor_w1, self.actor_w2 = action_update(self.actor_w1, self.actor_w2, critic_factor, 0.1*Ea, X, u, g)
       
                # Save history
                angle_hist.append(angle)
                vel_hist.append(X[0,3])
                j_hist.append(J[0,0])
                u_hist.append(u)
                x_hist.append(X[0,2])
                aw1_hist.append(np.mean(self.actor_w1))
                aw2_hist.append(np.mean(self.actor_w2))
                cw1_hist.append(np.mean(self.critic_w1))
                cw2_hist.append(np.mean(self.critic_w2))

                # Break the loop if we fail to keep the angle in range
                if r == -1:
                    fail = True
            
                    # Print a summary
                    print("Epoch:", '%04d' % (self.epoch+1), "max was:", '%06d' % (self.max_index + 1), "steps, this epoch was:", '%06d' % (i + 1))

                    # Save best run only
                    if i > max_i:
                        self.max_index = i
                        self.best_angle_hist = angle_hist
                        self.best_vel_hist   = vel_hist
                        self.best_j_hist     = j_hist  
                        self.best_u_hist     = u_hist
                        self.best_x_hist     = x_hist
                        self.best_aw1_hist   = aw1_hist
                        self.best_aw2_hist   = aw2_hist
                        self.best_cw1_hist   = cw1_hist
                        self.best_cw2_hist   = cw2_hist
        
                # Check if we reached the max time step
                if i == self.max_episode_steps:
                    self.trainingDone = True
                    print("Epoch:", '%04d' % (self.epoch+1), " MAX STEP COUNT REACHED, 600,000!")
    
                # Increment the time index and save variables
                i = i + 1
                Jlast = J
    


