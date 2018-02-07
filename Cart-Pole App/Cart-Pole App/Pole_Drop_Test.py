#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 07:58:34 2018

@author: David
"""
import numpy as np
import csv
from random import uniform
import StateModel as sm
import matplotlib.pyplot as plt

timestep = 500
angle_output = np.zeros(timestep)
angle_acc_output = np.zeros(timestep)

u = 0
dt = .02
state = np.array([uniform(-np.deg2rad(12),np.deg2rad(12)),0,0,0,0]).reshape(1,5) #[angle, ang_vel, dist, vel]
state = np.array([np.deg2rad(10),0,0,0,0]).reshape(1,5) #[angle, ang_vel, dist, vel]


with open('StateOutpus.csv', 'w') as csvfile:
    fieldnames = ['Iteration','Angle', 'Angle_Velocity','Angle_Accel', 'Distance', 'Velocity', 'Force', 'Nc']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for i in range(timestep):
        
        state = np.array(sm.cart_pole_model(dt,state,0,u)).reshape(1,5)
        print("Iteration:", '%04d' % (i+1), "force=", '%03d' % (u),"Ang:", "{:.3f}".format(state[0,0]), "AngVel:", "{:.3f}".format(state[0,1]), "Dist=", "{:.3f}".format(state[0,2]), "Vel=", "{:.3f}".format(state[0,3]))
        writer.writerow({'Iteration': str(i+1), 'Angle': "{:.3f}".format(np.rad2deg(state[0,0])), 'Angle_Velocity': "{:.3f}".format(np.rad2deg(state[0,1])), 'Angle_Accel': "{:.3f}".format(np.rad2deg(state[0,4])), 'Distance': "{:.3f}".format(state[0,2]), 'Velocity':"{:.3f}".format(state[0,3]), 'Force':'%03d' % (u)})
        angle_output[i] = np.rad2deg(state[0,0])%360
        angle_acc_output[i] = state[0,4]

plt.title("Pendulum angle over time", fontsize=14)
plt.plot(angle_output)
plt.ylabel("Pendulum Angle, deg")
plt.xlabel("Time Step")
plt.show()

plt.title("Pendulum acc over time", fontsize=14)
plt.plot(angle_acc_output)
plt.ylabel("Angular Acc, rad/s**2")
plt.xlabel("Time Step")
plt.show()