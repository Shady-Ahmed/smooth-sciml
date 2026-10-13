# -*- coding: utf-8 -*-
"""
Code accompanying the book chapter "Multifidelity Operator Learning" 
    Authors: Amanda A. Howard, Panos Stinis, and Shady E. Ahmed
    For comments or inquiries, please contact: shady.ahmed@pnnl.gov

This code is part of the chapter published in "An Introduction to Scientific
    Machine Learning: Mathematical Foundations and Numerical Implementation"

Description:
    Full order model (FOM) data generation for the 1D viscous Burgers problem 
    with an initial condition of a square wave
"""

#%% Import libraries
import os
import numpy as np
import sys
from utilities import pade4d, pade4dd

#%% Define Functions

###############################################################################
#compute rhs for numerical solutions
#  r = -u*u' + nu*u''
###############################################################################
def rhs(nx,dx,nu,u):
    r = np.zeros(nx+1)
    u2 = u**2
    up = pade4d(u, dx, nx)
    up2 = pade4d(u2,dx,nx)
    upp = pade4dd(u,dx,nx)

    r = -(u*up + up2)/3 + nu*upp 
    return r

#%% Creating folders
# data folder
if os.path.isdir("./data"):
    print('Data folder already exists')
else: 
    print('Creating data folder')
    os.makedirs("./data")

# plot folder
if os.path.isdir("./plots"):
    print('Plots folder already exists')
else: 
    print('Creating plots folder')
    os.makedirs("./plots")

#%% Main program:
    
quick_mode = "--quick" in sys.argv
if quick_mode:
    print("Running Example 5_1 in QUICK mode (generating Re = 5.0e3 and test Re = 15.0e3).")

# Inputs
nx =  4*1024  #spatial resolution
lx = 1.0    #spatial domain
dx = lx/nx
x = np.linspace(0, lx, nx+1)

#Reynolds number
re_train = [5.0e3] if quick_mode else [2.5e3, 5.0e3, 7.5e3, 10.0e3]
re_test = 15.0e3

#width of initial condition wave
xic = 0.5

tm = 0.1 if quick_mode else 1 #maximum time
      
dt = 1e-5   #solver timestep
nt = int(tm/dt) #number of timesteps
t = np.linspace(0, tm, nt+1)

ns = int(100*tm) #number of snapshots to save
freq = int(nt/ns) #frequency of data storing

#%% Training Dataset

for re in re_train:
    
    print("Generating " + str(ns) +" Snapshots of Training Data at Re = " + str(re))

    #initial condition
    uu = np.zeros(nx+1)
    uu[np.where(x<=xic)] = 1
    
    # boundary conditions: b.c. (not updated)
    u1 = np.zeros(nx+1)
    uu[0] = 0.0
    uu[nx] = 0.0
    u1[0] = 0.0
    u1[nx]= 0.0       

    #check for stability
    nu = 1/re   #control dissipation
    neu = nu*dt/(dx*dx)
    cfl = np.max(uu)*dt/dx
    if (neu >= 0.25):
        print('Neu condition: reduce dt')
        sys.exit()
    if (cfl >=  0.5):
        print('CFL condition: reduce dt')
        sys.exit()         
    
    uFOM = np.zeros([ns+1, nx+1])
    
    #time integration
    if1 = 0
    uFOM[0,:] = uu.copy()
    for jj in range(1,nt+1):
        #RK3 scheme -- RK4 can be used instead
        # first step
        rr = rhs(nx,dx,nu,uu)
        u1[1:nx] = uu[1:nx] + dt*rr[1:nx]
        
        # second step
        rr = rhs(nx,dx,nu,u1)
        u1[1:nx] = 0.75*uu[1:nx] + 0.25*u1[1:nx] + 0.25*dt*rr[1:nx]
        	
        # third step
        rr = rhs(nx,dx,nu,u1)
        uu[1:nx] = 1.0/3.0*uu[1:nx] + 2.0/3.0*u1[1:nx] + 2.0/3.0*dt*rr[1:nx]
               
        # store  velocity field
        if(np.mod(jj,freq) == 0):
            if1=if1+1
            uFOM[if1,:] = uu.copy() 
            print("Snapshot: {0:2d} -- Time: {1:4.2f}".format(if1, t[jj]))
    
        # check for CFL
        cfl = np.max(uu)*dt/dx
    
        #check for numerical stability
        if (cfl > 1.0):
            print('Error: CFL limit exceeded')
            break
        
    print('Saving training data')      
    np.save('./data/ufom-re-'+str(int(int(re)))+'-Train.npy',uFOM)
   
#%% Testing Dataset

print("Generating " + str(ns) +" Snapshots of Testing Data at Re = " + str(re_test))

#initial condition
uu = np.zeros(nx+1)
uu[np.where(x<=xic)] = 1

# boundary conditions: b.c. (not updated)
u1 = np.zeros(nx+1)
uu[0] = 0.0
uu[nx] = 0.0
u1[0] = 0.0
u1[nx]= 0.0   

#check for stability
nu = 1/re_test   #control dissipation
neu = nu*dt/(dx*dx)
cfl = np.max(uu)*dt/dx
if (neu >= 0.25):
    print('Neu condition: reduce dt')
    sys.exit()
if (cfl >=  0.5):
    print('CFL condition: reduce dt')
    sys.exit()
      
uFOM = np.zeros([ns+1, nx+1])

#time integration
if1 = 0
uFOM[0,:] = uu.copy()
for jj in range(1,nt+1):
    #RK3 scheme -- RK4 can be used instead
    # first step
    rr = rhs(nx,dx,nu,uu)
    u1[1:nx] = uu[1:nx] + dt*rr[1:nx]
    
    # second step
    rr = rhs(nx,dx,nu,u1)
    u1[1:nx] = 0.75*uu[1:nx] + 0.25*u1[1:nx] + 0.25*dt*rr[1:nx]
    	
    # third step
    rr = rhs(nx,dx,nu,u1)
    uu[1:nx] = 1.0/3.0*uu[1:nx] + 2.0/3.0*u1[1:nx] + 2.0/3.0*dt*rr[1:nx]
           
    # store  velocity field
    if(np.mod(jj,freq) == 0):
        if1=if1+1
        uFOM[if1,:] = uu.copy() 
        print("Snapshot: {0:2d} -- Time: {1:4.2f}".format(if1, t[jj]))

    # check for CFL
    cfl = np.max(uu)*dt/dx

    #check for numerical stability
    if (cfl > 1.0):
        print('Error: CFL limit exceeded')
        break
    
print('Saving testing data')      
np.save('./data/ufom-re-'+str(int(int(re_test)))+'-Test.npy',uFOM)
