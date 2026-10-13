# -*- coding: utf-8 -*-
"""
Code accompanying the book chapter "Multifidelity Operator Learning" 
    Authors: Amanda A. Howard, Panos Stinis, and Shady E. Ahmed
    For comments or inquiries, please contact: shady.ahmed@pnnl.gov

This code is part of the chapter published in "An Introduction to Scientific
    Machine Learning: Mathematical Foundations and Numerical Implementation"

Description:
    MFON training to learn closure
"""

#%% Import libraries

import os
import sys
import pickle

# linear algebra, optimization, and DL libraries
import jax
import jax.numpy as jnp
from jax import random
from jax.example_libraries import optimizers

import numpy as np
import matplotlib.pyplot as plt

from utilities import GROM_RK3, batched_GROM_RK3
from utilities import BatchGenerator, Normalizer, GROM_DeepONet_InTheLoop

#%% Extra platform settings

# Global flag to set a specific platform, must be used at startup.
# Defaults to CPU for portability across systems; switch to 'gpu' if a GPU device is present.
platform = 'gpu' if jax.default_backend() == 'gpu' else 'cpu'
jax.config.update('jax_platform_name', platform)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

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

#%% Main code:
# =============================================
# =============================================
if __name__ == "__main__":

    quick_mode = "--quick" in sys.argv

    # Inputs
    nx =  4*1024  #spatial resolution
    lx = 1.0    #spatial domain
    dx = lx/nx
    x  = np.linspace(0, lx, nx+1)
    
    # Reynolds number: in quick mode use 2 Re values, otherwise all 4
    # (or automatically match whatever training snapshots exist in data/)
    available_re = [rek for rek in [2.5e3, 5.0e3, 7.5e3, 10.0e3] if os.path.isfile(f'./data/ufom-re-{int(int(rek))}-Train.npy')]
    if quick_mode:
        re_train = [rek for rek in [2.5e3, 5.0e3] if rek in available_re] or available_re
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
    
    #%% Reading FOM snapshot data
    print('Reading FOM snapshots...')
 
    ufomlist_train = []
    ufomlistall = []

    for rek in re_train:
        uu = np.load('./data/ufom-re-'+str(int(int(rek)))+'-Train.npy')
        ufomlist_train.append(uu)
        ufomlistall.append(np.vstack(uu))
    ufomc = np.vstack(ufomlistall)    

    ufom_test = np.load('./data/ufom-re-'+str(int(int(re_test)))+'-Test.npy')
    
    #%% Reading POD and GROM data
    filename = "./data/pickled_grom_data_nr_" + str(nr) +"-Train"
    with open(filename, "rb") as fp:   #Unpickling
        pickled_data = pickle.load(fp)
        
    um = pickled_data['um']     
    phi = pickled_data['phi']
    atruelist_train = pickled_data['atruelist']
    agplist_train = pickled_data['agplist']
    b_clist_train = pickled_data['b_clist']
    b_llist_train = pickled_data['b_llist']
    b_nllist_train = pickled_data['b_nllist']
    grom_param_list_train = {'nr':nr, 'dt':dt,
                             'b_c':jnp.array(b_clist_train),
                             'b_l':jnp.array(b_llist_train),
                             'b_nl':jnp.array(b_nllist_train)}
            
    
    filename = "./data/pickled_grom_data_nr_" + str(nr) +"-Test"
    with open(filename, "rb") as fp:   #Unpickling
        pickled_data = pickle.load(fp)
        
    atrue_test = pickled_data['atrue']
    agp_test = pickled_data['agp']
    b_c_test = pickled_data['b_c']
    b_l_test = pickled_data['b_l']
    b_nl_test = pickled_data['b_nl']

    #%% IN THE LOOP
    print("============================ DeepONet  ============================")
            
    quick_mode = "--quick" in sys.argv
    train_flag = "--train" in sys.argv

    num_steps = 2 if quick_mode else 20
    # Enable training either by default or via flags
    train = True
    
    #% DeepONet Parameters
    dim_branch_input = nr+nr
    dim_trunk_input = 2
    dim_output = 1
    
    lr = 1e-3
    act_function = jnp.tanh
    
    N_nonlin = 10
    nonlin_branch_layers = [dim_branch_input, N_nonlin, N_nonlin, N_nonlin]
    nonlin_trunk_layers = [dim_trunk_input, N_nonlin, N_nonlin, N_nonlin]

    epochs = 100 if quick_mode else int(100e3) 
    batch_size = 32

    seed = 0

    if quick_mode:
        print(f"Running Example 5_3 in QUICK TEST mode (epochs={epochs}).")

    if train:
            
        # creating branch, trunk, output dataset
        n = np.arange(ns-num_steps+1)
        
        ubranch = []
        ytrunk = []
        ahigh = []
        
        for kk, rek in enumerate(re_train):
            atrue = atruelist_train[kk]
            ubranch.append(atrue[n,:])  
            
            ahigh_tmp = np.array([ np.array([atrue[i+j] for j in range(1,num_steps+1)]).T for i in n])
            ahigh.append(ahigh_tmp)
            
        ytrunk.append(jnp.repeat(np.hstack([jnp.arange(1,nr+1),jnp.array(re_train)]).reshape(1,-1), ns-num_steps+1, axis=0) )

        ubranch = jnp.hstack(ubranch)
        ytrunk = jnp.hstack(ytrunk)
        ahigh = jnp.hstack(ahigh)
        
        # here ubranch is arranged as follows: (ns-num_steps+1) x (nr*len(re_train))
        # [a1(t_0, re_1), a2(t_0, re_1), ..., a_nr(t_0, re_1), a1(t_0, re_2), a2(t_0, re_2), ...]
        # [a1(t_1, re_1), a2(t_1, re_1), ..., a_nr(t_1, re_1), a1(t_1, re_2), a2(t_1, re_2), ...]
        # ...
        # [a1(t_n, re_1), a2(t_n, re_1), ..., a_nr(t_n, re_1), a1(t_n, re_2), a2(t_n, re_2), ...]
        # ---------------------------------------------------------------------------------------
        # ytrunk is arranged as: (ns-num_steps+1) x (nr + len(re_train))
        # [1, 2, 3, ..., nr, re_1, re_2, re_3, ...]
        # [1, 2, 3, ..., nr, re_1, re_2, re_3, ...]
        # ...
        # [1, 2, 3, ..., nr, re_1, re_2, re_3, ...]
        # -----------------------------------------
        # ahigh is arranged as:  (ns-num_steps+1) x (nr*len(re_train)) x (num_steps)
        # ahigh[k] can be written as: (nr*len(re_train)) x (num_steps)
        # [a1(t_k+1, re_1), a1(t_k+2, re_1), ..., a1(t_num_steps, re_1)]
        # [a2(t_k+1, re_1), a2(t_k+2, re_1), ..., a2(t_num_steps, re_1)]
        # ...
        # [a_nr(t_k+1, re_1), a_nr(t_k+2, re_1), ..., a_nr(t_num_steps, re_1)]
        # [a1(t_k+1, re_2), a1(t_k+2, re_2), ..., a1(t_num_steps, re_2)]
        # [a2(t_k+1, re_2), a2(t_k+2, re_2), ..., a2(t_num_steps, re_1)]
        # ...
        # [a2(t_k+1, re_2), a2(t_k+2, re_2), ..., a2(t_num_steps, re_2)]
        # ...
        # --------------------------------------------------------------
        
        filename = "./data/ubranch-train.npy"
        jnp.save(filename,ubranch)
        
        filename = "./data/ytrunk-train.npy"
        jnp.save(filename,ytrunk)
                    
        filename = "./data/ahigh-train.npy"
        jnp.save(filename,ahigh)
                  
                
        key  = random.PRNGKey(seed)
        key, subkey = random.split(key)
        idx = random.choice(subkey, ubranch.shape[0], (int(ubranch.shape[0]),), replace=False)
        
        ubranch_train = ubranch[idx,:].copy()
        ytrunk_train = ytrunk[idx,:].copy()
        ahigh_train = ahigh[idx,:].copy()
        
        ubranch_normalizer, ytrunk_index_normalizer, ytrunk_re_normalizer, ahigh_normalizer = [Normalizer() for _ in range(4)]

        ubranch_train_split = jnp.vstack(jnp.split(ubranch_train,len(re_train),axis=1))  
        ubranch_normalizer.initialize(ubranch_train_split)
        ubranch_train_split = ubranch_normalizer.encode(ubranch_train_split)
        ubranch_train = jnp.hstack(jnp.split(ubranch_train_split,len(re_train),axis=0))
        
        ytrunk_index = ytrunk_train[0,:nr].flatten().copy()
        ytrunk_re = ytrunk_train[0,nr:].flatten().copy()
        ytrunk_index_normalizer.initialize(ytrunk_index)
        ytrunk_re_normalizer.initialize(ytrunk_re)
        for i in range(nr):
            ytrunk_train = ytrunk_train.at[:,i].set(ytrunk_index_normalizer.encode(ytrunk_train[:,i]))
        for k in range(len(re_train)):
            ytrunk_train = ytrunk_train.at[:,nr+k].set(ytrunk_re_normalizer.encode(ytrunk_train[:,nr+k]))
            
        ahigh_train_split = jnp.vstack(jnp.split(ahigh_train,len(re_train)*nr,axis=1))
        ahigh_normalizer.initialize(ahigh_train_split.reshape(-1,1))
        ahigh_train_split = ahigh_normalizer.encode(ahigh_train_split)
        ahigh_train = jnp.hstack(jnp.split(ahigh_train_split,len(re_train)*nr,axis=0))

        key, subkey = random.split(key)
        dataset = BatchGenerator(ubranch_train, ytrunk_train, ahigh_train, batch_size, subkey)
        
        key, subkey = random.split(key)
        model = GROM_DeepONet_InTheLoop(nonlin_branch_layers, nonlin_trunk_layers,
                                        act_function, ubranch_normalizer=ubranch_normalizer,
                                        ytrunk_index_normalizer=ytrunk_index_normalizer,
                                        ytrunk_re_normalizer=ytrunk_re_normalizer,ahigh_normalizer=ahigh_normalizer,
                                        batched_GROM_RK3=batched_GROM_RK3, grom_param_list=grom_param_list_train,
                                        key=subkey, lr=lr)           
            
        print('\n ... Training starting ...')
        model.train(dataset, nIter=epochs, num_steps=num_steps)        
        loss_training_log = model.loss_training_log
        loss_best_log = model.loss_best_log
        print('\n ... Training done ...')
        
        opt_state = model.opt_state
        unpacked_opt_state = optimizers.unpack_optimizer_state(opt_state)
        params = model.get_params(opt_state)

        pickled_data = {'unpacked_opt_state': unpacked_opt_state, 'params':params,
                        'loss_training_log': loss_training_log,
                        'loss_best_log' : loss_best_log,
                        'ubranch_normalizer': [ubranch_normalizer.mean,ubranch_normalizer.std,ubranch_normalizer.eps],
                        'ytrunk_index_normalizer':[ytrunk_index_normalizer.mean,ytrunk_index_normalizer.std,ytrunk_index_normalizer.eps],
                        'ytrunk_re_normalizer': [ytrunk_re_normalizer.mean,ytrunk_re_normalizer.std,ytrunk_re_normalizer.eps],
                        'ahigh_normalizer':[ahigh_normalizer.mean,ahigh_normalizer.std,ahigh_normalizer.eps]} 
        
        if not os.path.isdir("./models"):
            os.makedirs("./models")     #create data folder
        filename = "./models/DeepONet_model_num_steps_"+str(num_steps)+"_epochs_"+str(epochs)

        with open(filename, "wb") as fp:   #Pickling
            pickle.dump(pickled_data, fp)
        
    else:
        
        filename = "./models/DeepONet_model_num_steps_"+str(num_steps)+"_epochs_"+str(epochs)
        with open(filename, "rb") as fp:   #Unpickling
            pickled_data = pickle.load(fp)
        
        unpacked_opt_state = pickled_data['unpacked_opt_state']        
        params = pickled_data['params']
        opt_state = optimizers.pack_optimizer_state(unpacked_opt_state)

        loss_training_log = pickled_data['loss_training_log']        
        loss_best_log = pickled_data['loss_best_log']        

        ubranch_normalizer, ytrunk_index_normalizer, ytrunk_re_normalizer, ahigh_normalizer = [Normalizer() for _ in range(4)]
        ubranch_normalizer.import_normalizer(pickled_data['ubranch_normalizer'])
        ytrunk_index_normalizer.import_normalizer(pickled_data['ytrunk_index_normalizer'])
        ytrunk_re_normalizer.import_normalizer(pickled_data['ytrunk_re_normalizer'])
        ahigh_normalizer.import_normalizer(pickled_data['ahigh_normalizer'])
        
        model = GROM_DeepONet_InTheLoop(nonlin_branch_layers, nonlin_trunk_layers,
                                        act_function, ubranch_normalizer=ubranch_normalizer,
                                        ytrunk_index_normalizer=ytrunk_index_normalizer,
                                        ytrunk_re_normalizer=ytrunk_re_normalizer,ahigh_normalizer=ahigh_normalizer)  
        
    #%% Testing

    grom_param = {'nr':nr, 'dt':dt, 'b_c':b_c_test, 'b_l':b_l_test, 'b_nl':b_nl_test}
                                               
    ytrunk = jnp.hstack([jnp.arange(1,nr+1).reshape(-1,1), re_test*jnp.ones([nr,1])])  #tmp.reshape(-1,1)
    #encode to trunk scaling
    ytrunk = ytrunk.at[:,0].set(ytrunk_index_normalizer.encode(ytrunk[:,0]))
    ytrunk = ytrunk.at[:,1].set(ytrunk_re_normalizer.encode(ytrunk[:,1]))
                    
    apred = np.zeros_like(atrue_test)
    apred[0,:] = atrue_test[0,:].copy()
    for n in range(ns):
        acurrent = apred[n,:].copy()
        #GROM update
        a_next = GROM_RK3(acurrent,nr,dt,b_c_test,b_l_test,b_nl_test)
        #encode to branch scaling
        acurrent = ubranch_normalizer.encode(acurrent.reshape(1,-1))
        a_next = ubranch_normalizer.encode(a_next.reshape(1,-1))
        ubranch = jnp.repeat(jnp.hstack([acurrent,a_next]), nr, axis=0)
        
        #decode from branch scaling, then encode to output scaling
        a_next = ubranch_normalizer.decode(a_next.reshape(1,-1))
        a_next = ahigh_normalizer.encode(a_next.reshape(-1,))
            
        yp = a_next + model.predict(params,ubranch,ytrunk)
        
        #decode from output scaling
        apred[n+1,:] = ahigh_normalizer.decode(yp)

    #%% Plotting
    
    if os.path.isdir("./plots"):
        print('Plots folder already exists')
    else: 
        print('Creating plots folder')
        os.makedirs("./plots")


    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,4))
    ax = ax.flatten()
    for i, k in enumerate([0,1,8,9]):
    
        ax[i].plot(t, atrue_test[:,k], label=r'\bf{True}', color='C0', linewidth=2)
        ax[i].plot(t, agp_test[:,k], label=r'\bf{GPOD}', linestyle='-.', color='C1', linewidth=2)
        ax[i].plot(t, apred[:,k], label=r'\bf{MFON}', linestyle='--', color='C2', linewidth=2)
    
        ax[i].set_xlabel(r'$t$', fontsize=14, labelpad=5)
        ax[i].set_ylabel(r'$a_{'+str(k+1) +'}(t)$', fontsize=16, labelpad=5)

        ax[i].set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        
    ax[1].legend(loc="center", bbox_to_anchor=(1.2,1.15), ncol=3, fontsize=16, handletextpad=0.4, columnspacing=2)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    filename = "./plots/prom_re_"+str(int(re_test))+".png"
    plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
    filename = "./plots/prom_re_"+str(int(re_test))+".pdf"
    plt.savefig(filename, dpi = 300, bbox_inches = 'tight')