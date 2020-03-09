"""
Author:         David Beam, db4ai
Date:           18 January 2018
Description:    Configuration parameters to run the following models
                    Main_NN   - basic multilayer perceptron model
                    Main_NARX - nonlinear autoregressive mlp with exgenous inputs (delayed inputs and outputs fed back into the NN)
"""
import os
import datetime

# General NN parameters
trainingPct                 = 90            # The amount of data to use for training
epochs                      = 100          # Number of iterations or training cycles, includes both the FeedFoward and Backpropogation
learning_rate               = 0.0001          # Learning Rate 
dropout_output_keep_prob    = .75           # Percentage of neurons to keep between 
hidden_layer_widths         = [100]         # List of hidden layer widths (num neurons per hidden layer)
display_step                = 100           # How often to update the console with text
init_weights_bias_mean_val  = 0.0           # Mean value of the normal distribution used to initialize the weights and biases
init_weights_bias_std_dev   = 1.0           # Standard Deviation of the normal distribution used to initialize the weights and biases
alpha                       = .1            # cost-to-go discount factor

action_input_size           = 4             # Input parameter size for the action NN
critic_input_size           = 6             # Input parameter size for the critic NN

retrain_state               = 10            # The number of times to repeat the training cycle for each physical state update

# Cart Pole Physics Model parameters
dt = 0.02                                   # Time delta for each step
useOpenAImodel = True                       # Use the OpenAI cart pole model
renderOpenAImodel = True                    # Render a movie of the OpenAI progress, does not work in spyder gui


# Path settings, checks for Windows environment to choose between \ and /
if os.name == 'nt':
    dir_char = '\\'
else:
    dir_char = '/'
dir_path = os.path.dirname(os.path.realpath(__file__))
log_dir = dir_char + 'logs'