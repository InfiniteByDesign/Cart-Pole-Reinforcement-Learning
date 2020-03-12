#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import the librarys
from appJar.appjar import gui
import ActorCritic

# listen for buttons
def initialize(button):
    #ac.actor_width = 100
    #ac.critic_width = 100
    #ac.bias = 
    #ac.std_dev = 
    #ac.epochs = 
    #ac.retrain_state = 
    #ac.learning_rate = 
    #ac.alpha = 
    #ac.max_episode_steps = 
    ac.Set_Hyperparameters()
    ac.InitializeWeights()
    ac.InitializeHistory()

def trainInit(button):
    ac.Train_Init()

def trainStep(button):
    ac.Train_Step()

# create a GUI variable called app
app = gui()

# add & configure widgets - widgets get a name, to help referencing them later
app.addLabel("title", "Welcome to appJar")
app.setLabelBg("title", "red")
app.addButtons(["Initialize"], initialize)
app.addButtons(["Train Init"], trainInit)
app.addButtons(["Train"], trainStep)

# initilze objects
ac = ActorCritic.ActorCriticClass()

# start the GUI
app.go()


    