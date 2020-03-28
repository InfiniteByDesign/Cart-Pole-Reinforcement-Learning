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
# Misc Functions
# -------------------------------------------------------------------------

# The Norm1 function
def Norm1(vector):
    temp = 0
    for e in vector:
        temp = temp + np.abs(e)
    return temp

# ------------------------------------------------------------------------------------------------------------------------------------------
#
#   VARIABLE DESCRIPTIONS
#
#
#   Variables
#
#       X(t)    State variables from the Cart-Pole environment
#
#       u(t)    Actor network output
#       e_a(t)  Error of the actor network
#       E_a(t)  Objective function of the actor network
#
#       J(t)    Critic network output
#       e_c(t)  Error of the critic network
#       E_c(t)  Objective function of the critic network
#
#       U_c(t)  Heuristic term to balance the Bellman equation, 0 in our case
#       r(t)    Binary feedback, 0=success and keep going, 1=failed simulation
#       alpha   Discount factor for the infinite-horizon problem (0 to 0.95)
#
# ------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------
#
#       Actor Network
#
#       la      Learning rate of the actor network
#       4       Number of actor network inputs (4 state variables)
#       Na      Number of neurons in hidden layer
#       wa1     Input to hidden layer weights       [Na x 4]
#       wa2     Hidden to output layer weigths      [Na x 1]
#       x       Input to the actor network          [4  x 1]
#       x_cu    Input to the critic network from the actor network ( u(t) only )
#
#       Input to Hidden Layer
#             h(t)   =  wa1(t)  *  x(t)
#           [Na x 1] = [Na x 4] x [4 x 1]
#
#       Hidden Layer Activation
#             g(t)   = ( 1-e^-h(t) ) / ( 1+e^-h(t) ) 
#           [Na x 1] =    [Na x 1]  ./   [Na x 1]       
#           
#       Hidden Layer Output
#             v(t)   =  wa2(t)' *   g(t)
#            [1 x 1] = [1 x Na] x [Na x 1]
#
#       Output Layer Activation
#             u(t)   = ( 1-e^-v(t) ) / ( 1+e^-v(t) ) 
#            [1 x 1] =   [1 x 1]     /    [1 x 1]
#
#       Update wa2 (dwa2)
#           dwa2     = la * [ -dE_a / dwa2 ]
#           dwa2     = la * [ -dE_a / dJ ] * [ dJ / du ] * [ du / dv ] * [ dv / dwa2 ]
#           dwa2     = -la * e_a * [ 1/2(1 - u^2) ] *    g     * SUM(   wc2    *  [ 1/2(1 - p^2) ] * x_cu )
#           [Na x 1] = [1] x [1] x        [1]       x [Na x 1] x SUM( [Nc x 1] .x     [Nc x 1]     x [1]  )
#
#       Update wa1 (dwa1)
#           dwa1     = la * [ -dE_c / dw1 ]
#           dwa1     = la * [ -dE_a / dJ ] * [ dJ / du ] * [ du / dv ] * [ dv / dg ] * [ dg / dh ] * [ dh / dw1 ]
#           dwa1     = -la * e_a * [ 1/2(1 - u^2) ] *    wa2    *  [ 1/2(1 - g^2) ] *    x    * SUM(    wc2   *  [ 1/2(1 - p^2) ] * x_cu' )
#           [Na x 4] = [1] x [1] x        [1]       x [Na x 1] .x     [Na x 1]     x [1 x 4] x SUM( [Nc x 1] .x     [Nc x 1]     x  [1]  )
#
# ------------------------------------------------------------------------------------------------------------------------------------------
   
def actor_output(w1, w2, X):
    hidden  = np.dot(X,w1)
    g       = (1-np.exp(-hidden))/(1+np.exp(-hidden))
    v       = np.dot(g,w2)
    u       = (1-np.exp(-v))/(1+np.exp(-v))
    return u, g

def actor_cost(J):
    # Action Error:       e_a(t) = J(t) - U_c(t)
    # Objective Function: E_a(t) = 1/2 e_a(t)^2
    return 0.5 * J**2

def actor_update(actor_w1, actor_w2, critic_factor, error, X, u, g, actor_width, learning_rate): 
    # Change in w1 and w2
    d_w1 = -learning_rate * error * 0.5*(1-np.power(u,2)) * np.outer(np.multiply(actor_w2, (1 - np.power(g,2)).reshape(actor_width,1)), X) * critic_factor
    d_w2 = -learning_rate * error * 0.5*(1-np.power(u,2)) * g * critic_factor
    # Update the weights
    w1 = actor_w1 + np.transpose(d_w1)
    w2 = actor_w2 + np.transpose(d_w2)

    return w1, w2

# ------------------------------------------------------------------------------------------------------------------------------------------
#
#       Critic Network 
#
#       lc      Learning rate of the critic network
#       5       Number of critic network inputs (4 state variables + actor network output)
#       Nc      Number of neurons in hidden layer
#       wc1     Input to hidden layer weights      [Nc x 5]
#       wc2     Hidden to output layer weigths     [Nc x 1]
#       x       Input to the critic network        [5  x 1]
#
#       Input to Hidden Layer
#             q(t)   =  wc1(t)  *  x(t)
#           [Nc x 1] = [Nc x 5] x [5 x 1]
#
#       Hidden Layer Activation
#             p(t)   = ( 1-e^-q(t) ) / ( 1+e^-q(t) ) 
#           [Nc x 1] =    [Nc x 1]  ./   [Nc x 1]       
#           
#       Hidden Layer Output
#             J(t)   =  wc2(t)' *   p(t)
#            [1 x 1] = [1 x Nc] x [Nc x 1]
#
#       Update w2 (dwc2)
#           dwc2     = lc * [ -dE_c / dwc2 ]
#           dwc2     = lc * [ -dE_c / dJ ] * [ dJ / dwc2 ]
#           dwc2     = -lc * alpha * e_c * p
#           [Nc x 1] = [1] x  [1]  x [1] x [Nc x 1]
#
#       Update w1 (dwc1)
#           dwc1     = lc * [ -dE_c / dwc1 ]
#           dwc1     = lc * [ -dE_c / dJ ] * [ dJ / dp ] * [ dp / dq ] * [ dq / dwc1 ]
#           dwc1     = -lc * alpha * e_c *    wc2    *   [ 1/2(1 - p^2) ] *   x'
#           [Nc x 5] =  [1] x [1]  x [1] x [Nc x 1] .x     [Nc x 1]      x [1 x 5]
#
# ------------------------------------------------------------------------------------------------------------------------------------------

def critic_output(w1, w2, input):
    q = np.dot(input,w1)
    _p = (1-np.exp(-q))/(1+np.exp(-q))
    J = np.dot(_p,w2)
    return J, _p

def critic_cost(alpha, J, Jlast, r):
    # Prediction error:   e_c(t) = alpha * J(t) - [J(t-1)-r(t)]
    # Objective function: E_c(t) = 1/2 * e_c(t)^2
    return 0.5*(alpha*J - (Jlast-r))**2

def critic_update(critic_w1, critic_w2, error, x_a, _p, alpha, critic_width, critic_inputs, learning_rate):
    # Change in w2
    d_w2    = -learning_rate * (alpha * error * _p).reshape(critic_width,1)
    # Change in w1
    temp_a  = x_a.reshape(critic_inputs,1)
    temp_b  = -learning_rate * alpha * error * critic_w2 * (0.5*(1-np.power(_p,2).reshape(critic_width,1)))
    d_w1    = np.outer(temp_a,temp_b)
    # Update the weights
    w1 = critic_w1 + d_w1
    w2 = critic_w2 + d_w2
    return w1, w2

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
        self.actor_inputs       = 4 # Number of inputs to the Actor NN
        self.actor_width        = 0 # The number of hidden neurons in the single hidden layer, actor
        self.actor_cycle        = 0 # Number of times to retrain each time step
        self.actor_learn_rate   = 0 # Initial learning rate
        self.actor_lrate        = 0 # Current learning rate
        self.actor_learn_decay  = 0 # Learning rate decay each time step
        self.actor_learn_min    = 0 # Minimum learning rate
        self.actor_err_thresh   = 0 # Minimum error threshold, don't train below threshold

        self.critic_inputs      = 5 # Number of inputs to the Critic NN
        self.critic_width       = 0 # The number of hidden neurons in the single hidden layer, critic
        self.critic_cycle       = 0 # Number of times to retrain each time step
        self.critic_learn_rate  = 0 # Initial learning rate
        self.critic_lrate       = 0 # Current learning rate
        self.critic_learn_decay = 0 # Learning rate decay each time step
        self.critic_learn_min   = 0 # Minimum learning rate
        self.critic_err_thresh  = 0 # Minimum error threshold, don't train below threshold
        self.alpha              = 0 # cost-to-go discount factor

        self.lowlimit           = 0 # Lower limit of uniform distribution used to initialize the weights and biases
        self.highlimit          = 0 # Upper limit of uniform distribution used to initialize the weights and biases
        
        self.epochs             = 0 # Number of iterations or training cycles, includes both the FeedFoward and Backpropogation
        self.max_episode_steps  = 0 # Max number of trials before ending the experiment

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
        self.best_criticcost    = []
        self.latest_criticcost  = []

        # Training Flow
        self.trainingDone       = False
        self.epoch              = 0
        self.env                = gym.make('CartPole-v0')
        self.renderOpenAImodel  = True

        # Vector to normalize the evironment state
        #   Max angle = +/- 12
        #   Max angular velocity = +/- 120
        #   Max horizontal distance = +/- 2.4
        #   Max horizontal velocity = +/- 1.5
        self.stateNorm         = [12, 120, 2.4, 1.5 ]

    # Initialize Weights, there are 4 inputs to the actor and 6 inputs to the critic
    def InitializeWeights(self):
        self.actor_w1  = np.ones((self.actor_inputs,self.actor_width),dtype=float)  * np.random.uniform(-1,1,(self.actor_inputs,self.actor_width))
        self.actor_w2  = np.ones((self.actor_width,1),dtype=float)  * np.random.uniform(-1,1,(self.actor_width,1))
        self.critic_w1 = np.ones((self.critic_inputs,self.critic_width),dtype=float) * np.random.uniform(-1,1,(self.critic_inputs,self.critic_width))
        self.critic_w2 = np.ones((self.critic_width,1),dtype=float) * np.random.uniform(-1,1,(self.critic_width,1))

    # Initialize the history values
    def InitializeHistory(self):
        self.max_index          = 0
        self.best_criticcost    = []
        self.latest_criticcost  = []
        self.best_angle_hist    = []
        self.best_vel_hist      = []
        self.best_j_hist        = []
        self.best_u_hist        = []
        self.best_x_hist        = []
        self.best_aw1_hist      = []
        self.best_aw2_hist      = []
        self.best_cw1_hist      = []
        self.best_cw2_hist      = []

    # Get the last cost lists
    def GetCostHistory(self):
        return range(len(self.latest_criticcost)), self.latest_criticcost, range(len(self.best_criticcost)), self.best_criticcost

    # -------------------------------------------------------------------------
    # Set Hyperparameters
    # -------------------------------------------------------------------------

    def Set_Hyperparameters(self, actorWidth, actorCycle, actorlearningrate, actorLearnDecay, actorMinLearn, actorErrThreshold, criticWidth, criticCycle, criticlearningrate, criticLearnDecay, criticMinLearn, criticErrThreshold, alpha, lowlimit, highlimit, epochs, episodes, showOpenAI):
        
        self.actor_width        = actorWidth            # The number of hidden neurons in the single hidden layer, actor
        self.actor_cycle        = actorCycle            # Number of times to retrain each time step
        self.actor_learn_rate   = actorlearningrate     # Initial learning rate
        self.actor_learn_decay  = actorLearnDecay       # Learning rate decay each time step
        self.actor_learn_min    = actorMinLearn         # Minimum learning rate
        self.actor_err_thresh   = actorErrThreshold     # Minimum error threshold, don't train below threshold

        self.critic_width       = criticWidth           # The number of hidden neurons in the single hidden layer, critic
        self.critic_cycle       = criticCycle           # Number of times to retrain each time step
        self.critic_learn_rate  = criticlearningrate    # Initial learning rate
        self.critic_learn_decay = criticLearnDecay      # Learning rate decay each time step
        self.critic_learn_min   = criticMinLearn        # Minimum learning rate
        self.critic_err_thresh  = criticErrThreshold    # Minimum error threshold, don't train below threshold
        self.alpha              = alpha                 # cost-to-go discount factor

        self.lowlimit           = lowlimit              # Lower limit of uniform distribution used to initialize the weights and biases
        self.highlimit          = highlimit             # Upper limit of uniform distribution used to initialize the weights and biases
        
        self.epochs             = epochs                # Number of iterations or training cycles, includes both the FeedFoward and Backpropogation
        self.max_episode_steps  = episodes              # Max number of trials before ending the experiment
        self.renderOpenAImodel  = showOpenAI            # Render a movie of the OpenAI progress, does not work in spyder gui

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    # Initialize values before a new training sequence
    def Train_Init(self):
        self.trainingDone   = False
        self.max_index      = 0
        self.epoch          = 0
        self.actor_lrate    = self.actor_learn_rate
        self.critic_lrate   = self.critic_learn_rate
                   
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
        self.latest_criticcost = []
        angle_hist  = []
        vel_hist    = []
        j_hist      = []
        u_hist      = []
        x_hist      = []
        aw1_hist    = []
        aw2_hist    = []
        cw1_hist    = []
        cw2_hist    = []
        success_string = ""

        # Continue training if we are not done
        if self.epoch < self.epochs and self.trainingDone != True:
            # initialize values for this training step
            self.epoch = self.epoch+1
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
                
                # Increment the setp index
                i = i + 1
    
                # Action
                u, g = actor_output(self.actor_w1, self.actor_w2, X/self.stateNorm)
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
                # state = [ang, ang vel, dist, vel, ang_acc] from the OpenAI Cart-Pole Environment
                # X = [dist, vel, angle, angular velocity] 
                state, reward, done, info = self.env.step(action_u)
                X = np.array([np.rad2deg(state[2]),np.rad2deg(state[3]),state[0],state[1]]).reshape(1,4)

                # Determine if the episode failed before we normalize the state vector
                angle = X[0,0]%360
                if angle > 180: angle = angle - 360
                fail = not (angle <= 12 and angle >= -12 and X[0,2]>-2.4 and X[0,2]<2.4)
                r = (0, -1)[fail]

                # Critic, create the critic input and evaluate the network
                critic_input    = np.concatenate((X/self.stateNorm,np.array([u],dtype=float).reshape((1,1))),axis=1)
                J, _p           = critic_output(self.critic_w1,self.critic_w2,critic_input)
        
                # Update the critic network
                Ec = critic_cost(self.alpha, J, Jlast, r)
                numUpdates = 0
                while Ec > self.critic_err_thresh and numUpdates<self.critic_cycle:
                    # Update the weights then calculate the error again
                    self.critic_w1, self.critic_w2 = critic_update(self.critic_w1, self.critic_w2, Ec, critic_input, _p, self.alpha, self.critic_width, self.critic_inputs, self.critic_lrate)
                    J, _p = critic_output(self.critic_w1,self.critic_w2,critic_input)
                    Ec = critic_cost(self.alpha, J, Jlast, r)
                    # Increase the update count
                    numUpdates = numUpdates + 1

                # Update the actor network
                Ea = actor_cost(J)
                numUpdates = 0
                while Ea > self.actor_err_thresh and numUpdates<self.actor_cycle:
                    # Update the weights
                    critic_factor = np.sum(0.5 * np.multiply(np.multiply(np.transpose(self.critic_w2),(1-np.power(_p,2))),self.critic_w1[4,:]))
                    self.actor_w1, self.actor_w2 = actor_update(self.actor_w1, self.actor_w2, critic_factor, Ea, X/self.stateNorm, u, g, self.actor_width, self.actor_lrate)
                    # Calculate the critic output based on the actor command so we can get a new J value                    
                    u, g = actor_output(self.actor_w1, self.actor_w2, X/self.stateNorm)
                    critic_input    = np.concatenate((X/self.stateNorm,np.array([u],dtype=float).reshape((1,1))),axis=1)
                    J, _p = critic_output(self.critic_w1,self.critic_w2,critic_input)
                    Ea = actor_cost(J)
                    # Increase the update count
                    numUpdates = numUpdates + 1

                # Normalize the weights if they are getting too large
                if np.max(self.actor_w1)  > 1.5: self.actor_w1  = self.actor_w1/np.max(self.actor_w1)
                if np.max(self.actor_w2)  > 1.5: self.actor_w2  = self.actor_w2/np.max(self.actor_w2)
                if np.max(self.critic_w1) > 1.5: self.critic_w1 = self.critic_w1/np.max(self.critic_w1)
                if np.max(self.critic_w2) > 1.5: self.critic_w2 = self.critic_w2/np.max(self.critic_w2)
       
                # Decrease the learning rates every 5 steps
                if self.epoch%5 == 0: 
                    self.actor_lrate = self.actor_lrate - self.actor_learn_decay
                    self.critic_lrate = self.critic_lrate - self.critic_learn_decay

                # Cap the minimum learning rate
                if self.actor_lrate < self.actor_learn_min: self.actor_lrate = self.actor_learn_min
                if self.critic_lrate < self.critic_learn_decay: self.critic_lrate = self.critic_learn_decay

                # Save history
                self.latest_criticcost.append(Ec)
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
                    success_string = "Failed"
                    fail = True
            
                    # Print a summary
                    print("Epoch:", '%04d' % (self.epoch), "max was:", '%06d' % (self.max_index), "steps, this epoch was:", '%06d' % i)

                # Check if we reached the max time step
                if i == self.max_episode_steps:
                    success_string = "Success"
                    self.trainingDone = True
                    print("Epoch:", '%04d' % (self.epoch), " MAX STEP COUNT REACHED!")
                
                # Training is done if we have reached the maximum number of trials (epochs)
                if self.epoch == self.epochs:
                    self.trainingDone = True

                # Save best run only
                if i > self.max_index:
                    self.max_index = i
                    self.best_criticcost = self.latest_criticcost
                    self.best_angle_hist = angle_hist
                    self.best_vel_hist   = vel_hist
                    self.best_j_hist     = j_hist  
                    self.best_u_hist     = u_hist
                    self.best_x_hist     = x_hist
                    self.best_aw1_hist   = aw1_hist
                    self.best_aw2_hist   = aw2_hist
                    self.best_cw1_hist   = cw1_hist
                    self.best_cw2_hist   = cw2_hist

                # Save the last critic cost
                Jlast = J
    
            # This training sequence has failed or has finished, return the results
            return self.epoch, i, self.max_index, success_string, self.trainingDone, self.actor_width, self.actor_cycle, self.actor_learn_rate, self.actor_lrate, self.actor_learn_decay, self.actor_learn_min, self.actor_err_thresh, self.critic_width, self.critic_cycle, self.critic_learn_rate, self.critic_lrate, self.critic_learn_decay, self.critic_learn_min, self.critic_err_thresh, self.alpha, self.lowlimit, self.highlimit, self.epochs, self.max_episode_steps


