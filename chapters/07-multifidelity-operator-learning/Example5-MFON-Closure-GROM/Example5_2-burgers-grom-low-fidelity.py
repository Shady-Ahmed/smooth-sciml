# -*- coding: utf-8 -*-
"""
Code accompanying the book chapter "Multifidelity Operator Learning" 
    Authors: Amanda A. Howard, Panos Stinis, and Shady E. Ahmed
    For comments or inquiries, please contact: shady.ahmed@pnnl.gov

This code is part of the chapter published in "An Introduction to Scientific
    Machine Learning: Mathematical Foundations and Numerical Implementation"

Description:
    Galerkin POD ROM for 1D Burgers problem
"""

#%% Import libraries
import os
import sys
import numpy as np
import pickle 

from utilities import pod, pod_proj
from utilities import const_term,  lin_term, nonlin_term, GROM_RK3, batched_GROM_RK3
from jax import vmap

#%% format plots
import matplotlib as mpl
import shutil
# Check if LaTeX is available on the system; fall back gracefully to standard fonts if not
if shutil.which("latex") is not None:
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath} \boldmath"
else:
    mpl.rc('text', usetex=False)
font = {'weight' : 'bold',
        'size'   : 20}
mpl.rc('font', **font)

#%% Main program:
# =============================================
if __name__ == "__main__":
# =============================================

    quick_mode = "--quick" in sys.argv
    if quick_mode:
        print("Running Example 5_2 in QUICK mode.")

    # Inputs
    nx =  4*1024  #spatial resolution
    lx = 1.0    #spatial domain
    dx = lx/nx
    x  = np.linspace(0, lx, nx+1)
    
    # Reynolds number: in quick mode use 2 Re values, otherwise all 4
    # (or automatically match whatever training snapshots exist in data/)
    available_re = [rek for rek in [2.5e3, 5.0e3, 7.5e3, 10.0e3] if os.path.isfile(f'./data/ufom-re-{int(int(rek))}-Train.npy')]
    if quick_mode:
        re_train = [rek for rek in [5.0e3] if rek in available_re] or available_re
    else:
        re_train = available_re if available_re else [2.5e3, 5.0e3, 7.5e3, 10.0e3]
    re_test = 15.0e3

    xic = 0.5
    
    tm = 0.1 if quick_mode else 1     #maximum time
    nt = 10 if quick_mode else 100   #number of timesteps
    ns = 10 if quick_mode else 100   #number of snapshot
    dt = tm/nt
    freq = int(nt/ns)

    t = np.linspace(0, tm, ns+1)

    nr = 10     #number of modes
    
    #%% FOM snapshot generation for training
    print('Reading FOM snapshots...')
    ufomlist_train = []
    ufomlistall = []

    for rek in re_train:
        uu = np.load('./data/ufom-re-'+str(int(int(rek)))+'-Train.npy')
        ufomlist_train.append(uu)
        ufomlistall.append(np.vstack(uu))
    ufomc = np.vstack(ufomlistall)    

    ufom_test = np.load('./data/ufom-re-'+str(int(int(re_test)))+'-Test.npy')[:ns+1]
        
    #%% POD basis computation for training data
    print('Computing POD basis...')
    phi = np.zeros([nx+1,nr]) #POD modes     
    eigv = np.zeros(ns+1) #Eigenvalues     
    
    # mean subtraction
    um = np.mean(ufomc,0)
    phi, eigv, ric  = pod(ufomc - um.reshape(1,-1), nr) 
    
    #%% Calculating true POD modal coefficients
    
    print('Computing true POD coefficients...')
    batched_pod_proj = vmap(pod_proj, in_axes=(0, None))
       
    atruelist_train = []
    for kk, rek in enumerate(re_train):
        ufom = ufomlist_train[kk]
        atruelist_train.append(batched_pod_proj(ufom-um.reshape(1,-1),phi) )
    
    atrue_test = batched_pod_proj(ufom_test-um.reshape(1,-1),phi)
        
    #%% Galerkin projection precomputed coefficients
    print('Computing GP coefficients...')
    b_clist_train = [const_term(nr,um,phi,rek,nx,dx) for rek in re_train]
    b_llist_train = [lin_term(nr,um,phi,rek,nx,dx) for rek in re_train]
    b_nllist_train = [nonlin_term(nr,phi,rek,nx,dx) for rek in re_train]
    
    b_c_test = const_term(nr,um,phi,re_test,nx,dx)
    b_l_test = lin_term(nr,um,phi,re_test,nx,dx) 
    b_nl_test = nonlin_term(nr,phi,re_test,nx,dx)
    
    #%% Solving ROM
    
    #=========================================================================#
    print("Solving GROM For Training Parameters")
    #=========================================================================#
    
    agplist_train = []
    agp_single_steplist_train = []
    for kk, rek in enumerate(re_train):
        
        b_c = b_clist_train[kk]
        b_l = b_llist_train[kk]
        b_nl = b_nllist_train[kk]
        
        agp_tmplist = []
        agp_single_tmplist = []

        atrue = atruelist_train[kk]
        agp_single_steplist_train.append(batched_GROM_RK3(atrue[:-1,],nr,dt,b_c,b_l,b_nl))

        agp = np.zeros([ns+1,nr])
        ag = atrue[0,:].copy()
        agp[0,:] = np.copy(ag)
        time = 0
        for k in range(1,nt+1):
            time = np.round(time+dt,decimals=5)
            ag = GROM_RK3(ag,nr,dt,b_c,b_l,b_nl) #solving GROM without closure 
            if (k%freq == 0):
                agp[int(k/freq),:] = np.copy(ag)
         
        agplist_train.append(agp)
        print("Re= "+str(rek)+ "-- Done")

    pickled_data = {'um': um, 'phi':phi, 'atruelist': atruelist_train,
                    'agplist': agplist_train, 'agp_single_steplist':agp_single_steplist_train,
                    'b_clist': b_clist_train, 'b_llist':b_llist_train, 'b_nllist':b_nllist_train}
    
    filename = "./data/pickled_grom_data_nr_" + str(nr) +"-Train"
    with open(filename, "wb") as fp:   #Pickling
        pickle.dump(pickled_data, fp)
    
    del agplist_train, agp_single_steplist_train, pickled_data

    
    #=========================================================================#
    print("Solving GROM For Testing Parameters")
    #=========================================================================#
    
    agp_single_step_test = batched_GROM_RK3(atrue_test[:-1,],nr,dt,b_c_test,b_l_test,b_nl_test)
       
    agp_test = np.zeros([ns+1,nr])
    ag = np.copy(atrue_test[0,:])
    agp_test[0,:] = np.copy(ag)
    time = 0
    for k in range(1,nt+1):
        time = np.round(time+dt,decimals=5)
        # Solve GROM without closure using test Reynolds number coefficients
        ag = GROM_RK3(ag,nr,dt,b_c_test,b_l_test,b_nl_test)
        if (k%freq == 0):
            agp_test[int(k/freq),:] = np.copy(ag)

    print("Re= "+str(re_test) + "-- Done")

    pickled_data = {'um': um, 'phi':phi, 'atrue': atrue_test,
                    'agp': agp_test, 'agp_single_step':agp_single_step_test,
                    'b_c': b_c_test, 'b_l':b_l_test, 'b_nl':b_nl_test}
    
    filename = "./data/pickled_grom_data_nr_" + str(nr) +"-Test"
    with open(filename, "wb") as fp:   #Pickling
        pickle.dump(pickled_data, fp)
    
    del agp_test, agp_single_step_test, pickled_data
