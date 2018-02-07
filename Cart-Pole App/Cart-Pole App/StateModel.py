"""
    Physics model for the cart-pole balancing NN 
"""
import numpy as np 
from numpy.linalg import inv

def cart_pole_model(dt,x,flag,u):
    # Constants/System Properties
    g               = 9.8
    l               = 0.5
    mc              = 1.0
    mp              = 0.1
    uc              = 0.0005
    up              = 0.02
    
    # Inputs:
    F               = u
    
    # States:
    angle           = x[0,0]
    ang_vel         = x[0,1]
    dist            = x[0,2]
    x_vel           = x[0,3]
    ang_acc         = x[0,4]
    
    
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
    
    # Angular Updates
    new_ang_vel     = ang_vel + new_ang_acc * dt
    angle           = angle + ang_vel*dt + 0.5*new_ang_acc*dt**2
    angle           = angle# % (2*np.pi)
    
    # New Linear Accelleration
    new_x_acc = (F + mp*l*((new_ang_vel**2)*np.sin(angle) - new_ang_acc*np.cos(angle)) - uc*Nc*sgn) / (mc + mp)
    
    # Linear Updates
    new_x_vel       = x_vel + new_x_acc * dt
    dist            = (new_x_vel**2 - x_vel**2)/(2*new_x_acc) + dist

    return [angle,new_ang_vel,dist,new_x_vel,new_ang_acc]
    
    """
    # Equations from JenniSi Paper
    sgn = 0
    if x_vel > 0:
        sgn = 1
    elif x_vel < 0:
        sgn = -1
    
    # New Angular Accelleration
    temp            = -F-mp*l*ang_vel**2*np.sin(angle) + uc*sgn   
    new_ang_acc_num = g*np.sin(angle) + np.cos(angle)*temp - up*ang_vel/(mp*l)    
    new_ang_acc_den = l*(4/3 - (mp*np.cos(angle)**2/(mc+mp)))   
    new_ang_acc     = new_ang_acc_num / new_ang_acc_den
    
    # Angular Updates
    new_ang_vel     = ang_vel + new_ang_acc * dt
    angle           = angle + ang_vel*dt + 0.5*new_ang_acc*dt**2
    angle           = angle % (2*np.pi)
    
    # New Linear Accelleration
    new_x_acc = (F + mp*l*(new_ang_vel**2*np.sin(angle) - new_ang_acc**2*np.cos(angle)) - uc*sgn) / (mc + mp)
    
    # Linear Updates
    new_x_vel       = x_vel + new_x_acc * dt
    dist            = (new_x_vel**2 - x_vel**2)/(2*new_x_acc) + dist
    
    return [angle,new_ang_vel,dist,new_x_vel,new_ang_acc]
    """