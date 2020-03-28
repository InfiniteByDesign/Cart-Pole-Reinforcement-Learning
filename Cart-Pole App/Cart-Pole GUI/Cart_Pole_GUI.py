#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import the librarys
from appJar.appjar import gui
import ActorCritic
import numpy as np
import datetime
import csv

mess1 = "Set initial hyperparameters, then press 'Apply/Initialize'"
mess2 = "Run a training sequence by pressing one of the 'Train' buttons"
mess3 = "When all trials are complete, you must 'Apply/Initialize' before restarting training"
mess4 = "Running all trials, wait for trials to complete"
mess5 = "All trials are complete, you must 'Apply/Initialize' before restarting training"

# Write results to a CSV file
def WriteResults(epoch, maxsteps, success_string, actor_width, actor_cycle, actor_learn_rate, actor_lrate, actor_learn_decay, actor_learn_min, actor_err_thresh, critic_width, critic_cycle, critic_learn_rate, critic_lrate, critic_learn_decay, critic_learn_min, critic_err_thresh, alpha, lowlimit, highlimit, epochs, max_episode_steps):
    # Create the output string
    outpuline = [epoch, maxsteps, success_string, actor_width, actor_cycle, actor_learn_rate, actor_lrate, actor_learn_decay, actor_learn_min, actor_err_thresh, critic_width, critic_cycle, critic_learn_rate, critic_lrate, critic_learn_decay, critic_learn_min, critic_err_thresh, alpha, lowlimit, highlimit, epochs, max_episode_steps]
    # Open the csv file and write the data
    with open("Cart-Pole Results.csv", mode='a') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(outpuline)

# listen for buttons
def trainInit(button):
    
    # Get Actor Parameters
    actorWidth          = int(app.getEntry("actorWidth"))
    actorCycle          = int(app.getEntry("actorCycle"))
    actorlearningrate   = app.getEntry("actorlearningrate")
    actorLearnDecay     = app.getEntry("actorLearnDecay")
    actorMinLearn       = app.getEntry("actorMinLearn")
    actorErrThreshold   = app.getEntry("actorErrThreshold")
        
    # Get Critic Parameters
    criticWidth         = int(app.getEntry("criticWidth"))
    criticCycle         = int(app.getEntry("criticCycle"))
    criticlearningrate  = app.getEntry("criticlearningrate")
    criticLearnDecay    = app.getEntry("criticLearnDecay")
    criticMinLearn      = app.getEntry("criticMinLearn")
    criticErrThreshold  = app.getEntry("criticErrThreshold")
    alpha               = app.getEntry("alpha")

    # Get other Parameters
    lowlimit            = app.getEntry("lowlimit")
    highlimit           = app.getEntry("highlimit")
    epochs              = int(app.getEntry("epochs"))
    episodes            = int(app.getEntry("episodes"))
    showOpenAI          = app.getCheckBox("showOpenAI")

    # Initialize and set the hyperparameters
    ac.Set_Hyperparameters(actorWidth, actorCycle, actorlearningrate, actorLearnDecay, actorMinLearn, actorErrThreshold, criticWidth, criticCycle, criticlearningrate, criticLearnDecay, criticMinLearn, criticErrThreshold, alpha, lowlimit, highlimit, epochs, episodes, showOpenAI)
    ac.Train_Init()

    ac.InitializeWeights()
    ac.InitializeHistory()
    app.setMessage("message", mess2)

def trainStep(button):
    epoch, steps, maxsteps, success_string, trainingDone, actor_width, actor_cycle, actor_learn_rate, actor_lrate, actor_learn_decay, actor_learn_min, actor_err_thresh, critic_width, critic_cycle, critic_learn_rate, critic_lrate, critic_learn_decay, critic_learn_min, critic_err_thresh, alpha, lowlimit, highlimit, epochs, max_episode_steps = ac.Train_Step()

    # Converting datetime object to string
    timestampStr = (datetime.datetime.now()).strftime("%d-%b-%Y (%H:%M:%S)")
    app.setLabel("2_LastRun", "Last Run: " + str(timestampStr))
    app.setLabel("2_LastEpoch", "Last Epoch: " + str(epoch))
    app.setLabel("2_LastSteps", "Last # of Steps: " + str(steps))
    app.setLabel("2_BestSteps", "Best # of Steps: " + str(maxsteps))
    app.setLabel("2_SuccessString", "Result: " + str(success_string))
    # Update the plots
    stepsc, cCost, stepsb, cCostBest = ac.GetCostHistory()
    axes = app.updatePlot("CriticCost", stepsc, np.squeeze(cCost))
    axes = app.updatePlot("BestCriticCost", stepsb, np.squeeze(cCostBest))
    if trainingDone == True:
        app.setMessage("message", mess5)
        WriteResults(epoch, maxsteps, success_string, actor_width, actor_cycle, actor_learn_rate, actor_lrate, actor_learn_decay, actor_learn_min, actor_err_thresh, critic_width, critic_cycle, critic_learn_rate, critic_lrate, critic_learn_decay, critic_learn_min, critic_err_thresh, alpha, lowlimit, highlimit, epochs, max_episode_steps)
    else:
        app.setMessage("message", mess3)

def trainAll(button):    
    # Loop through all epochs until done or success    
    app.setMessage("message", mess3)
    while ac.epoch < ac.epochs and ac.trainingDone != True:
        # Train the next step
        epoch, steps, maxsteps, success_string, trainingDone, actor_width, actor_cycle, actor_learn_rate, actor_lrate, actor_learn_decay, actor_learn_min, actor_err_thresh, critic_width, critic_cycle, critic_learn_rate, critic_lrate, critic_learn_decay, critic_learn_min, critic_err_thresh, alpha, lowlimit, highlimit, epochs, max_episode_steps = ac.Train_Step()
        # Update the results on the output tab
        timestampStr = (datetime.datetime.now()).strftime("%d-%b-%Y (%H:%M:%S)")
        app.setLabel("2_LastRun", "Last Run: " + str(timestampStr))
        app.setLabel("2_LastEpoch", "Last Epoch: " + str(epoch))
        app.setLabel("2_LastSteps", "Last # of Steps: " + str(steps))
        app.setLabel("2_BestSteps", "Best # of Steps: " + str(maxsteps))
        app.setLabel("2_SuccessString", "Result: " + str(success_string))
        # Update the plots
        stepsc, cCost, stepsb, cCostBest = ac.GetCostHistory()
        axes = app.updatePlot("CriticCost", stepsc, np.squeeze(cCost))
        axes = app.updatePlot("BestCriticCost", stepsb, np.squeeze(cCostBest))
        if trainingDone == True:
            app.setMessage("message", mess5)
            WriteResults(epoch, maxsteps, success_string, actor_width, actor_cycle, actor_learn_rate, actor_lrate, actor_learn_decay, actor_learn_min, actor_err_thresh, critic_width, critic_cycle, critic_learn_rate, critic_lrate, critic_learn_decay, critic_learn_min, critic_err_thresh, alpha, lowlimit, highlimit, epochs, max_episode_steps)
        else:
            app.setMessage("message", mess4)

def setupTab1(gui):
    gui.startTab("Setup")
    
    row = 0
    gui.addLabel("l2", "Hyperparameters", row, 0, 4); row = row+1
    gui.setLabelBg("l2", "green")

    gui.startLabelFrame("Actor NN Parameters")
    gui.addLabel("l3a", "Hidden Nodes", row, 0);            gui.addNumericEntry("actorWidth", row, 1);
    gui.addLabel("l3b", "Internal Cycle", row, 2);          gui.addNumericEntry("actorCycle", row, 3);              row=row+1
    gui.addLabel("l3c", "Learning Rate", row, 0, 1);        gui.addNumericEntry("actorlearningrate", row, 1, 2);   
    gui.addLabel("l3d", "Learning Rate Decay", row, 2);     gui.addNumericEntry("actorLearnDecay", row, 3);         row=row+1
    gui.addLabel("l3e", "Minimum Learn Rate", row, 0);      gui.addNumericEntry("actorMinLearn", row, 1);
    gui.addLabel("l3f", "Train Err Threshold", row, 2);     gui.addNumericEntry("actorErrThreshold", row, 3);       row=row+1
    gui.setEntry("actorWidth", "24")
    gui.setEntry("actorCycle", "100")
    gui.setEntry("actorlearningrate", "0.3")
    gui.setEntry("actorLearnDecay", "0.05")
    gui.setEntry("actorMinLearn", "0.005")
    gui.setEntry("actorErrThreshold", "0.005")
    gui.stopLabelFrame()
    
    gui.startLabelFrame("Critic NN Parameters")
    gui.addLabel("l4a", "Hidden Nodes", row, 0);            gui.addNumericEntry("criticWidth", row, 1);
    gui.addLabel("l4b", "Internal Cycle", row, 2);          gui.addNumericEntry("criticCycle", row, 3);             row=row+1
    gui.addLabel("l4c", "Learning Rate", row, 0, 1);        gui.addNumericEntry("criticlearningrate", row, 1, 2);   
    gui.addLabel("l4d", "Learning Rate Decay", row, 2);     gui.addNumericEntry("criticLearnDecay", row, 3);        row=row+1    
    gui.addLabel("l4e", "Minimum Learn Rate", row, 0);      gui.addNumericEntry("criticMinLearn", row, 1);
    gui.addLabel("l4f", "Train Err Threshold", row, 2);     gui.addNumericEntry("criticErrThreshold", row, 3);      row=row+1
    gui.addLabel("l4g", "Cost Horizon", row, 0);            gui.addNumericEntry("alpha", row, 1);                   row=row+1
    gui.setEntry("criticWidth", "24")
    gui.setEntry("criticCycle", "50")
    gui.setEntry("criticlearningrate", "0.3")
    gui.setEntry("criticLearnDecay", "0.05")
    gui.setEntry("criticMinLearn", "0.005")
    gui.setEntry("criticErrThreshold", "0.05")
    gui.setEntry("alpha", "0.0001")
    gui.stopLabelFrame()

    gui.startLabelFrame("NN Weights Initialization - Uniform Distribution")
    gui.addLabel("l5a", "Lower Limit", row, 0);            gui.addNumericEntry("lowlimit", row, 1)
    gui.addLabel("l5b", "Upper Limit", row, 2);         gui.addNumericEntry("highlimit", row, 3);                   row=row+1
    gui.setEntry("lowlimit", "-1.0")
    gui.setEntry("highlimit", "1.0")
    gui.stopLabelFrame()

    gui.startLabelFrame("Flow Control")
    gui.addLabel("l6a", "Total Num Trials", row, 0);        gui.addNumericEntry("epochs", row, 1); 
    gui.addLabel("l6b", "Num of Steps to Success", row, 2); gui.addNumericEntry("episodes", row, 3);                row=row+1
    gui.addNamedCheckBox("Show Cart-Pole Animation", "showOpenAI", row, 0);                                         row=row+1
    gui.setEntry("epochs", "10")
    gui.setEntry("episodes", "200")
    gui.setCheckBox("showOpenAI")
    gui.stopLabelFrame()
    
    gui.stopTab()

def setupTab2(gui):
    gui.startTab("Output")
    row = 0
    gui.addMessage("title", """Actor-Critic Cart-Pole Results""", row, 0, 2)
    gui.setMessageWidth("title", 400)
    gui.addHorizontalSeparator(colour="red")
    gui.addLabel("2_LastRun", "Last Run: Not yet run")
    gui.addLabel("2_LastEpoch", "Last Epoch: ")
    gui.addLabel("2_LastSteps", "Last # of Steps: ")
    gui.addLabel("2_BestSteps", "Best # of Steps: ")  
    app.addLabel("2_SuccessString", "Result: ")  
    gui.addEmptyLabel("2_e1")  
    gui.addEmptyLabel("2_e2")  
    gui.addEmptyLabel("2_e3")  
    gui.addEmptyLabel("2_e4")  
    gui.addEmptyLabel("2_e5")  
    gui.addEmptyLabel("2_e6")
    gui.stopTab()

def setupTab3(gui):
    gui.startTab("Critic Cost")
    axes = app.addPlot("CriticCost", 0, 0)
    gui.stopTab()

def setupTab4(gui):
    app.startTab("Best Critic Cost")
    axes = app.addPlot("BestCriticCost", 0, 0)
    app.stopTab()

# Create a GUI variable called app
app = gui("Actor-Critic Cart Pole Example", "800x700")
app.setFont("8")

# Create Tabs
app.startTabbedFrame("TabbedFrame")
app.setTabbedFrameTabExpand("TabbedFrame")
setupTab1(app)
setupTab2(app)
setupTab3(app)
setupTab4(app)
app.stopTabbedFrame()

row = 10
app.addLabel("l1", "Commands", row, 0, 2)
app.setLabelBg("l1", "green")

app.addButtons(["Apply Hyperparameters and Initialize"], trainInit)
app.addButtons(["Train: Step-by-Step"], trainStep)
app.addButtons(["Train: All Epochs"], trainAll)
app.addMessage("message", mess1)
app.setMessageWidth("message", 550)

# initilze objects
ac = ActorCritic.ActorCriticClass()

# start the GUI
app.go()