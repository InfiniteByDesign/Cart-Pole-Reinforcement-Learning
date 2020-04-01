----------------------------------------------------------------------------

    Actor-Critic Cart Pole Implementation

        By David Beam - djb302@msstate.edu

----------------------------------------------------------------------------

This implementation is from the paper:

    J. Si and Y. T. Wang, "On-line learning control by association and reinforcement," IEEE Trans. Neural Networks, vol. 12, no. 2, pp. 264-276, 2001.

----------------------------------------------------------------------------

Packages used in the development of this program.

    Python      3.6.4
    Numpy       1.14.0
    MatPlotLib  2.1.2
    datetime
    csv
    AppJar      http://appjar.info/
    OpenAI Gym  https://gym.openai.com/
    
    Notes about running this application.
        Create a python environment with similar versions of the packages above.  The AppJar package is distributed with this file and is already in the correct subdirectory.
        OpenAI Gym must be installed in the python environment, directions are located on the OpenAI Gym website.

----------------------------------------------------------------------------

Using this program

    This program implements the Actor-Critic RL algorithm in the referenced paper above.  A GUI is provided by running the file Cart_Pole_GUI.py
    
    Upon running, a command line window is used for displaying errors and diagnostics and a GUI displayed to allow the user to execute the algorithm and modify hyperparameters.
    
    The GUI is separated into two sections, the tabbed section on the top 3/4 of the page and 3 buttons on the bottom 1/4 of the page.  The various tabs are discussed below.
    
    The main workflow is as follows:
    
        1. Choose the hyperparameters by modifying the values on the SETUP tab.
        2. Press the "Apply Hyperparameters and Initialize" button to write the hyperparameters to the Actor-Critic model
        3. Initiate training through one of two ways
            3a. "Train: Step-by-Step" initiates a single trial where the algorithm runs until the trial is deemed a success or failure
            3b. "Train: All Trials" initiates a continuous training sequence until a trial is deemed a success or the maximum number of trials have failed.

        The results from each trial are stored in a CSV file named Cart-Pole Results.csv
        This file must already exist in the main directory. Trial results will be appended to the end of the file.  Should you need to recreate the file, the header is as follows:
        
            Last Trial Number,Max Number of Steps,Result,Actor Width,Actor Train Cycles,Actor Init Learn Rate,Actor Last Learn Rate,Actor Learn Rate Decay,Actor Min Learn Rate,Actor Error Threshold,Critic Width,Critic Train Cycles,Critic Init Learn Rate,Critic Last Learn Rate,Critic Learn Rate Decay,Critic Min Learn Rate,Critic Error Threshold,Discount Factor,Init Weights Low Limit,Init Weights High Limit,Max number of Trials,Steps to Success
        
----------------------------------------------------------------------------

The following is a description of the different tabs available in the GUI

    SETUP

        This tab allows the user to alter the hyperparameters prior to starting a sequence of trials.

        Actor NN Parameters

            Hidden Nodes            - The number of neurons in the single hidden layer
            Learning Rate           - Initial learning rate
            Learning Rate Decay     - Amount to decrease the learning rate each time step
            Minimum Learning Rate   - Minimum learning rate
            Internal Cycle          - Number of times to retrain the network each time step
            Train Error Threshold   - Minimum error threshold below which ceases the internal cycle training each time step

        Critic NN Parameters

            Hidden Nodes            - The number of neurons in the single hidden layer
            Learning Rate           - Initial learning rate
            Learning Rate Decay     - Amount to decrease the learning rate each time step
            Minimum Learning Rate   - Minimum learning rate
            Internal Cycle          - Number of times to retrain the network each time step
            Train Error Threshold   - Minimum error threshold below which ceases the internal cycle training each time step
            Cost Horizon            - The discount factor used in training the critic neural network

        NN Weights Initialization

            Lower Limit             - Lower limit of uniform distribution used to initialize the weights and biases
            Upper Limit             - Upper limit of uniform distribution used to initialize the weights and biases

        Flow Control

            Total Number of Trials  - Number of failed trials allowed before simulation stops
            Num of Steps to Success - Number of time steps required to declare a trial as successful
            
    OUTPUT
        
        This tab displays text results from each trial.  If the "Train: All Epochs" button is used, this screen will update after every trial while the training commences.
        
    CRITIC COST
    
        The most recent plot of the critic network cost function output versus step number.
        
    BEST CRITIC COST
    
        The best critic cost results from the entire training run, only updated when the current trial achieves a higher step number than all previous trials.
        
    PLOT RESULTS
    
        This tab allows the user to create a quick plot using a single dependent and independant variable.  The data is read from the CSV results file each time a plot is requested.
    
----------------------------------------------------------------------------

Additional Information: The following is copied from the ActorCritic.py file and describes the variables and equations used in the Actor and Critic models      

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
