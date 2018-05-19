#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 07:58:34 2018

@author: David
"""
import numpy as np
import csv
from random import uniform
import StateModel2 as sm

u = 0
dt = .02
state = np.array([175,0,0,0,0,0,0,0,0,0]).reshape(1,10) #[angle, ang_vel, dist, vel, F, a_num_f, a_num, a_den, x_acc, sgn]


with open('StateOutpus.csv', 'w') as csvfile:
    fieldnames = ['Iteration','Angle', 'Angle_Velocity','Angle_Accel', 'Distance', 'Velocity', 'Force', 'A_Num_F', 'A_Num', 'A_Den', 'X_Acc', 'Sgn']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for i in range(500):
        
        state = np.array(sm.cart_pole_model(dt,state,0,u)).reshape(1,10)
        print("Iteration:", '%04d' % (i+1), "force=", '%03d' % (u),"Ang:", "{:.3f}".format(state[0,0]), "AngVel:", "{:.3f}".format(state[0,1]), "Dist=", "{:.3f}".format(state[0,2]), "Vel=", "{:.3f}".format(state[0,3]))
        writer.writerow({'Iteration': str(i+1), 'Angle': "{:.3f}".format(state[0,0]), 'Angle_Velocity': "{:.3f}".format(state[0,1]), 'Angle_Accel': "{:.3f}".format(state[0,4]), 'Distance': "{:.3f}".format(state[0,2]), 'Velocity':"{:.3f}".format(state[0,3]), 'Force':'%03d' % (u), 'A_Num_F':"{:.3f}".format(state[0,5]), 'A_Num':"{:.3f}".format(state[0,6]), 'A_Den':"{:.3f}".format(state[0,7]), 'X_Acc':"{:.3f}".format(state[0,8]), 'Sgn':"{:.3f}".format(state[0,9])})