"""
    Physics model for the cart-pole balancing NN 
"""
import numpy as np 
from numpy.linalg import inv

def cart_pole_model(dt,angle,ang_vel,ang_acc,x,x_vel,u):
    # Runge Kutta equations        
    #k1 = cart_pole_model_delta(0.5*dt,angle,ang_vel,x_vel,x,u)
    k1 = [angle,ang_vel,ang_acc,x,x_vel]
    k2 = cart_pole_model_delta(dt/2, angle+dt*k1[0]/2,ang_vel+dt*k1[1]/2,ang_acc+dt*k1[2]/2,x+dt*k1[3]/2,x_vel+dt*k1[4]/2, u)
    k3 = cart_pole_model_delta(dt/2, angle+dt*k2[0]/2,ang_vel+dt*k2[1]/2,ang_acc+dt*k2[2]/2,x+dt*k2[3]/2,x_vel+dt*k2[4]/2, u)
    k4 = cart_pole_model_delta(dt  , angle+dt*k3[0]  ,ang_vel+dt*k3[1]  ,ang_acc+dt*k3[2]  ,x+dt*k3[3]  ,x_vel+dt*k3[4]  , u)
 
    # Update next value of each variable    
    angle   = angle   + (1.0/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    ang_vel = ang_vel + (1.0/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    ang_acc = ang_acc + (1.0/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    x       = x       + (1.0/6.0)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
    x_vel   = x_vel   + (1.0/6.0)*(k1[4] + 2*k2[4] + 2*k3[4] + k4[4])
    
    # Return the results
    return [angle,ang_vel,ang_acc,x,x_vel]

def cart_pole_model_delta(dt,angle,ang_vel,ang_acc,x_vel,x,F):
    # Constants/System Properties
    g               = 9.8
    l               = 0.5
    mc              = 1.0
    mp              = 0.1
    uc              = 0.0005
    up              = 0.00002
        
    
    # Equations from Correct equations for the dynamics of the
    #    cart-pole system
    #    Razvan V. Florian
    #    Center for Cognitive and Neural Studies (Coneural)
    #    Str. Saturn 24, 400504 Cluj-Napoca, Romania
    #    florian@coneural.org
    #    July 11, 2005; updated February 10, 2007
    #
    #    https://coneural.org/florian/papers/05_cart_pole.pdf
    
    Nc = (mc + mp)*g - mp*l*(ang_acc*np.sin(angle) + (ang_vel**2)*np.cos(angle))
    
    sgn = 0
    if Nc*x_vel > 0:
        sgn = 1
    elif Nc*x_vel < 0:
        sgn = -1
    
    # New Angular Accelleration
    temp            = ( -F-mp*l*(ang_vel**2)*( np.sin(angle) + uc*sgn*np.cos(angle) ))/(mc+mp) + uc*g*sgn    
    new_ang_acc_num = g*np.sin(angle) + np.cos(angle)*temp - up*ang_vel/(mp*l)    
    new_ang_acc_den = l*(4/3 - mp*np.cos(angle)*(np.cos(angle)-uc*sgn)/(mc+mp))    
    new_ang_acc     = new_ang_acc_num / new_ang_acc_den
    
    # New Linear Accelleration
    new_x_acc = (F + mp*l*((ang_vel**2)*np.sin(angle) - new_ang_acc*np.cos(angle)) - uc*Nc*sgn) / (mc + mp)

    ang_vel     = dt * new_ang_acc
    angle       = dt * ang_vel
    x_vel       = dt * new_x_acc
    x           = dt * x_vel + x

    return [dt*angle,dt*ang_vel,dt*new_ang_acc,dt*x_vel,dt*x]