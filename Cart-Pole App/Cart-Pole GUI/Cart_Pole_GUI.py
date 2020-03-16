#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import the librarys
from appJar.appjar import gui
import ActorCritic

# listen for buttons
def initialize(button):
    ac.Set_Hyperparameters()
    ac.InitializeWeights()
    ac.InitializeHistory()

def trainInit(button):
    ac.Train_Init()

def trainStep(button):
    ac.Train_Step()

# create a GUI variable called app
app = gui("Actor-Critic Cart Pole Example", "400x200")

# add & configure widgets - widgets get a name, to help referencing them later
app.addLabel("l1", "Initialization", 0, 0, 3)
app.setLabelBg("l1", "green")
app.addButtons(["Initialize Weights/History"], initialize, 1, 0)
app.addButtons(["Reset Training"], trainInit, 1, 2)


app.addLabel("l2", "Hyperparameters", 2, 0, 3)
app.setLabelBg("l2", "green")

app.addLabel("l3", "Actor NN Width", 3, 0, 1)
app.addNumericEntry("actorWidth", 3, 1, 2)

app.addLabel("l4", "Critic NN Width", 4, 0, 1)
app.addNumericEntry("criticWidth", 4, 1, 2)

app.addLabel("l5", "Weights Bias", 5, 0, 1)
app.addNumericEntry("bias", 5, 1, 2)

app.addLabel("l6", "Weights Std Dev", 6, 0, 1)
app.addNumericEntry("stddev", 6, 1, 2)

app.addLabel("l7", "Num of Episodes", 7, 0, 1)
app.addNumericEntry("episodes", 7, 1, 2)

app.addLabel("l8", "Num of Epochs", 8, 0, 1)
app.addNumericEntry("epochs", 8, 1, 2)

app.addLabel("l9", "Retrain # per Epoch", 9, 0, 1)
app.addNumericEntry("retrains", 9, 1, 2)

app.addLabel("l10", "Learning Rate", 10, 0, 1)
app.addNumericEntry("learningrate", 10, 1, 2)

app.addLabel("l11", "Alpha", 11, 0, 1)
app.addNumericEntry("alpha", 4, 1, 2)


app.setEntryDefault("actorWidth", "100")
app.setEntryDefault("criticWidth", "100")
app.setEntryDefault("bias", "0.0")
app.setEntryDefault("stddev", "1.0")
app.setEntryDefault("episodes", "600000")
app.setEntryDefault("epochs", "100")
app.setEntryDefault("retrains", "10")
app.setEntryDefault("learningrate", "0.0001")
app.setEntryDefault("alpha", "0.1")


app.addButtons(["Train"], trainStep, 12, 0, 1)

# initilze objects
ac = ActorCritic.ActorCriticClass()

# start the GUI
app.go()


    