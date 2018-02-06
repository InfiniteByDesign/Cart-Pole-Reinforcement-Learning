"""
    Physics model for the cart-pole balancing NN 
"""
import numpy as np

def cart_pole_model(t,x,flag,u):
    # Constants/System Properties
    g               = 9.8
    length          = 0.5
    m_cart          = 1.0
    m_pend          = 0.1
    friction_cart   = 0.0005
    friction_pend   = 0.000002
    phi = 0
    
    # Inputs:
    force           = u
    
    # States:
    angle           = x[0,0]
    angular_vel     = x[0,1]
    distance        = x[0,2]
    vel             = x[0,3]
    
    # Equations:
    total_m         = m_cart + m_pend
    momt_pend       = m_pend * length
    
    hforce          = momt_pend * (angular_vel**2) * np.sin(angle)
    part_num        = -force - hforce + friction_cart * np.sign(vel)
    
    denom_ang_vel   = length * (4/3 - m_pend * ((np.cos(angle))**2) / total_m)
    num_ang_vel     = g * np.sin(angle) * np.cos(phi) + np.cos(angle) * part_num / total_m - friction_pend * angular_vel / momt_pend
    
    dxdt1           = angular_vel
    dxdt2           = num_ang_vel / denom_ang_vel
    
    num_vel         = force-total_m * g * np.sin(phi) + hforce - momt_pend * dxdt2 * np.cos(angle) - friction_cart * np.sign(vel)
    
    dxdt3           = vel
    dxdt4           = num_vel / total_m
    
    # Output:
    dx              = [dxdt1,dxdt2,dxdt3,dxdt4]
    return dx