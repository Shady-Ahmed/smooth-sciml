#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code accompanying the book chapter "Multifidelity Operator Learning" 
    Authors: Amanda A. Howard, Panos Stinis, and Shady E. Ahmed
    For comments or inquiries, please contact: shady.ahmed@pnnl.gov

This code is part of the chapter published in "An Introduction to Scientific
    Machine Learning: Mathematical Foundations and Numerical Implementation"

Description:
    Utility functions for Galerkin POD models and MFON training
"""

#%% Import libraries
import os

# linear algebra, optimization, and DL libraries
import numpy as np
import jax
from jax import numpy as jnp
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree
from functools import partial
from tqdm import trange
import itertools

#%% Extra platform settings

# Global flag to set a specific platform, must be used at startup.
# more details: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
jax.config.update('jax_platform_name', 'cpu') #change to 'gpu', if available
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

#%%
###############################################################################
# Numerical Routines
###############################################################################
# Thomas algorithm for solving tridiagonal systems:    
def tdma(a, b, c, r, up, s, e):
    for i in range(s+1,e+1):
        b[i] = b[i] - a[i]/b[i-1]*c[i-1]
        r[i] = r[i] - a[i]/b[i-1]*r[i-1]    
    up[e] = r[e]/b[e]   

    for i in range(e-1,s-1,-1):
        up[i] = (r[i]-c[i]*up[i+1])/b[i]

# Computing first derivatives using the fourth order compact scheme:  
def pade4d(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    ud = np.zeros(n+1)
    i = 0
    b[i] = 1.0
    c[i] = 2.0
    r[i] = (-5.0*u[i] + 4.0*u[i+1] + u[i+2])/(2.0*h)
    
    a[1:n] = 1.0
    b[1:n] = 4.0
    c[1:n] = 1.0
    r[1:n] = 3.0*(u[2:n+1] - u[0:n-1])/h
   
    i = n
    a[i] = 2.0
    b[i] = 1.0
    r[i] = (-5.0*u[i] + 4.0*u[i-1] + u[i-2])/(-2.0*h)
    tdma(a, b, c, r, ud, 0, n)
    return ud
    
# Computing second derivatives using the foruth order compact scheme:  
def pade4dd(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    udd = np.zeros(n+1)
    i = 0
    b[i] = 1.0
    c[i] = 11.0
    r[i] = (13.0*u[i] - 27.0*u[i+1] + 15.0*u[i+2] - u[i+3])/(h*h)

    a[1:n] = 0.1
    b[1:n] = 1.0
    c[1:n] = 0.1
    r[1:n] = 1.2*(u[2:n+1] - 2.0*u[1:n] + u[0:n-1])/(h*h)
    
    i = n
    a[i] = 11.0
    b[i] = 1.0
    r[i] = (13.0*u[i] - 27.0*u[i-1] + 15.0*u[i-2] - u[i-3])/(h*h)
    
    tdma(a, b, c, r, udd, 0, n)
    return udd

#%%
###############################################################################
#POD Routines
###############################################################################         
def pod(u,nr): #Basis Construction
    ns,n = u.shape
    U,s,Vh = np.linalg.svd(u.T, full_matrices=False)
    phi = U[:,:nr]
    eigv = s**2
    #compute RIC (relative inportance index)
    ric = jnp.cumsum(eigv)/jnp.sum(eigv)*100   
    return phi,eigv,ric

def pod_proj(u, phi): #Projection
    a = jnp.dot(phi.T, u)  # u = Phi * a ==> a = Phi^T * u
    return a

def pod_rec(a, phi): #Reconstruction    
    u = jnp.dot(phi, a)   # u = Phi * a
    return u

#%%

###############################################################################
# Galerkin ROM Routines
###############################################################################

def GROMrhs(grom_param, a): 
    
    b_c = grom_param['b_c']
    b_l = grom_param['b_l']
    b_nl = grom_param['b_nl']           
   
    #constant term
    r1 = b_c
    
    #linear term
    r2 = jnp.dot(b_l, a)
    
    #nonlinear term
    r3 = jnp.sum(jnp.dot(b_nl, a)*a, axis=1)

    r = r1 + r2 + r3
    return r

# time integration using third-order Runge Kutta method
def GROM_RK3(a,nr,dt,b_c,b_l,b_nl):
    
    grom_param = {'nr':nr, 'b_c':b_c, 'b_l':b_l, 'b_nl':b_nl}

    c1 = 1.0/3.0
    c2 = 2.0/3.0
    
    #stage-1
    r = GROMrhs(grom_param, a)
    a0 = a + dt*r

    #stage-2
    r = GROMrhs(grom_param, a0)
    a0 = 0.75*a + 0.25*a0 + 0.25*dt*r

    #stage-3
    r = GROMrhs(grom_param, a0)
    a = c1*a + c2*a0 + c2*dt*r

    return a

batched_GROM_RK3 = vmap(GROM_RK3, in_axes=(0,None,None,None,None,None))

# GROM RHS term (projection of FOM operators onto ROM basis functions)
def const_term(nr, um, phi, re, nx, dx):

    umd = pade4d(um,dx,nx)     #first derivative of um    
    umdd = pade4dd(um,dx,nx)   #second derivative of um    
    
    # constant term
    b_c = np.zeros(nr)
    for k in range(nr):
        tmp = -um*umd + (1/re)*umdd
        b_c[k] = np.dot(tmp.T, phi[:,k])
            
    return b_c

def lin_term(nr, um, phi, re, nx, dx):
    
    umd = pade4d(um,dx,nx)     #first derivative of um    

    phid = np.zeros([nx+1,nr])
    phidd = np.zeros([nx+1,nr])
    for i in range(nr):
        phid[:,i] = pade4d(phi[:,i],dx,nx)   #first derivative of phi    
        phidd[:,i] = pade4dd(phi[:,i],dx,nx) #second derivative of phi    

    # linear term 
    b_l = np.zeros([nr,nr])
    for k in range(nr):
        for i in range(nr):
            tmp = -um*phid[:,i] - phi[:,i]*umd + (1/re)*phidd[:,i]
            b_l[k,i] = np.dot(tmp.T , phi[:,k]) 

    return b_l

def nonlin_term(nr, phi, re, nx ,dx):
    
    phid = np.zeros([nx+1,nr])
    for i in range(nr):
        phid[:,i] = pade4d(phi[:,i], dx, nx) #first derivative of phi  

    # nonlinear term 
    b_nl = np.zeros([nr,nr,nr])    
    for k in range(nr):
        for i in range(nr):
            for j in range(nr):
                tmp = phi[:,i]*phid[:,j]
                b_nl[k,i,j] = - np.dot( tmp.T, phi[:,k] ) 

    return b_nl

#%%
###############################################################################
# ANN Routines
###############################################################################  

def dataloader(arrays, batch_size, *, key):
    """Infinite generator yielding random mini-batches from the given arrays."""
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    if batch_size > dataset_size:
        batch_size = dataset_size
    while True:
        perm = jax.random.permutation(key, indices)
        (key,) = jax.random.split(key, 1)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            batch_arrays = tuple(array[batch_perm] for array in arrays)
            # Yield ((ubranch, ytrunk), ahigh) matching the expected batch signature
            yield ((batch_arrays[0], batch_arrays[1]), batch_arrays[2])
            start = end
            end = start + batch_size

class BatchGenerator:
    """Pure JAX iterable batch generator (drop-in replacement for torch Dataset)."""
    def __init__(self, ubranch, ytrunk, ahigh,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        self.ubranch = ubranch
        self.ytrunk = ytrunk
        self.ahigh = ahigh
        self.batch_size = batch_size
        self.key = rng_key

    def __iter__(self):
        return dataloader((self.ubranch, self.ytrunk, self.ahigh), self.batch_size, key=self.key)
                 
class Normalizer():
    def initialize(self, x, eps=0.000001):
        self.mean = jnp.mean(x, axis=0)
        self.std = jnp.std(x, axis=0)
        self.eps = eps
    def import_normalizer(self,params):
        mean,std,eps = params
        self.mean = mean
        self.std = std
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x
    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

#%%
class GROM_DeepONet_InTheLoop:     
    
    # Initialize the class
    def __init__(self, nonlin_branch_layers, nonlin_trunk_layers,
                       activation, ubranch_normalizer=None, ytrunk_index_normalizer=None,
                       ytrunk_re_normalizer=None,ahigh_normalizer=None,    
                       batched_GROM_RK3=None, grom_param_list=None,
                       key=random.PRNGKey(0), lr=1e-3): 
        
        
        if nonlin_branch_layers[-1] != nonlin_trunk_layers[-1]:
            raise ValueError("Output sizes of function NN and location NN do not match for the nonlinear DeepONet.")
       
        self.ubranch_normalizer, self.ytrunk_index_normalizer, self.ytrunk_re_normalizer, self.ahigh_normalizer \
           = ubranch_normalizer,      ytrunk_index_normalizer,      ytrunk_re_normalizer,      ahigh_normalizer

        self.grom_rk3 = batched_GROM_RK3        
        if batched_GROM_RK3 != None:
           self.grom_rk3_rebatch = vmap(batched_GROM_RK3,in_axes=(0,None,None,0,0,0))
        self.grom_param_list = grom_param_list

        self.nonlin_init, self.single_nonlin_apply = self.standard_deeponet(nonlin_branch_layers, nonlin_trunk_layers, activation=activation)
        self.nonlin_apply = vmap(self.single_nonlin_apply, in_axes=(None, 0, 0))
        self.nonlin_apply_rebatch = vmap(self.nonlin_apply, in_axes=(None, 0, 0))

        nonlinkey, key = random.split(key, num=2)
        params = self.nonlin_init(nonlinkey)
        
        self.params = params
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(lr, 
                                                                      decay_steps=5000, 
                                                                      decay_rate=0.97))
        self.opt_state = self.opt_init(params)
        
        self.best_opt_state = self.opt_state

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()

        # building loss function
        self.loss_best_log = []
        self.loss_training_log = []
        self.loss_testing_log = []

        
    def standard_deeponet(self, branch_layers, trunk_layers, activation=jnp.tanh):

        def xavier_init_j(key, d_in, d_out):
            glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(key, (d_out, d_in))
            b = jnp.zeros(d_out)
            return W, b
        
        def init(parent_key):
            
            def init_layer(key, d_in, d_out):
                k1, k2 = random.split(key)
                W, b = xavier_init_j(k1, d_in, d_out)
                return W, b
            
            branch_key, trunck_key = random.split(parent_key, num=2)

            branch_subkeys = random.split(branch_key, num=len(branch_layers)-1)
            trunk_subkeys = random.split(trunck_key, num=len(branch_layers)-1)

            branch_params = list(map(init_layer, branch_subkeys, branch_layers[:-1], branch_layers[1:]))
            trunk_params = list(map(init_layer, trunk_subkeys, trunk_layers[:-1], trunk_layers[1:]))
            return (branch_params, trunk_params)
                 
            
        #@partial(jit, static_argnums=(3,))
        def apply(params, ubranch, ytrunk):
            
            branch_params, trunk_params = params
            
            branch_hidden_layers = branch_params[:-1]
            trunk_hidden_layers = trunk_params[:-1]

            for w_branch, b_branch in branch_hidden_layers:
                ubranch = activation(jnp.dot(w_branch, ubranch) + b_branch)

            w_branch_last, b_branch_last = branch_params[-1]
            ubranch = jnp.dot(w_branch_last, ubranch) + b_branch_last

            for w_trunk, b_trunk in trunk_hidden_layers:                                
                ytrunk = activation(jnp.dot(w_trunk, ytrunk) + b_trunk)
                                        
            w_trunk_last, b_trunk_last = trunk_params[-1]
            ytrunk  = jnp.dot(w_trunk_last, ytrunk) + b_trunk_last

            outputs = jnp.sum(ubranch*ytrunk)
            
            return outputs
        

        return init, apply

           
    def loss(self, params, batch):
        
            
        (ubranch, ytrunk), ahigh = batch
       
        grom_param_list = self.grom_param_list
        num_steps = self.num_steps

        nr = grom_param_list['nr']
        dt = grom_param_list['dt']
        b_c_split = grom_param_list['b_c']
        b_l_split = grom_param_list['b_l']
        b_nl_split = grom_param_list['b_nl']      
        nre = b_c_split.shape[0]
        
        acurrent_split = jnp.array(jnp.split(ubranch,nre,axis=1))        
           
        loss = 0
        for n in range(num_steps):

            #decode from branch scaling to apply GROM update, then encode back to branch scaling
            acurrent_split_dec = jnp.array( [self.ubranch_normalizer.decode(acurrent_split[i]) for i in range(nre)]  )            
            apred_split = self.grom_rk3_rebatch(acurrent_split_dec,nr,dt,b_c_split,b_l_split,b_nl_split)
            apred_split = jnp.array( [self.ubranch_normalizer.encode(apred_split[i]) for i in range(nre)]  )
            ubranch_now = jnp.dstack([acurrent_split,apred_split])

            #decode from branch scaling, then encode into output scaling
            apred_split = jnp.array( [self.ubranch_normalizer.decode(apred_split[i]) for i in range(nre)]  )
            apred = jnp.hstack(apred_split)
            apred_one_column = jnp.vstack(jnp.split(apred,nre*nr,axis=1))
            apred_one_column = self.ahigh_normalizer.encode(apred_one_column)
            apred = jnp.hstack(jnp.split(apred_one_column,nre*nr,axis=0))
            apred_split =  jnp.array(jnp.split(apred,nre,axis=1))
                    
    
            for i in range(nr):
                ytrunk_i = jnp.array([ytrunk[:,[i,nr+kk]] for kk in range(nre)])
                
                nonlin_pred = self.nonlin_apply_rebatch(params, ubranch_now, ytrunk_i)
                apred_split = apred_split.at[:,:,i].add(nonlin_pred)

            #decode from output scaling, then encode to branch scaling
            apred = jnp.hstack(apred_split)
            apred_one_column = jnp.vstack(jnp.split(apred,nre*nr,axis=1))
            apred_one_column = self.ahigh_normalizer.decode(apred_one_column)
            apred = jnp.hstack(jnp.split(apred_one_column,nre*nr,axis=0))
            acurrent_split =  jnp.array(jnp.split(apred,nre,axis=1))
            acurrent_split = jnp.array( [self.ubranch_normalizer.encode(acurrent_split[i]) for i in range(nre)] )
            
            apred_local = jnp.hstack(apred_split).copy()
            ahigh_local = ahigh[:,:,n].copy()
            if apred_local.shape == ahigh_local.shape:
                loss += jnp.mean(( apred_local.flatten() -  ahigh_local.flatten() )**2)
            else:
                raise ValueError(f"Incompatible array shapes: apred_local has shape {apred_local.shape} and ahigh_local has shape {ahigh_local.shape}")
                
        loss /= num_steps
        return loss

       
    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state,  batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, batch)
        opt_state = self.opt_update(i, g, opt_state)    
        return opt_state


    # Optimize parameters in a loop
    def train(self, dataset, nIter = 1000, num_steps=1):
        
        self.num_steps = num_steps

        data = iter(dataset)

        pbar = trange(nIter)
        
        #initiating loss
        batch= next(data) # Fetch data
        params = self.get_params(self.opt_state)
        loss_best = self.loss(params, batch)
        self.best_opt_state = self.opt_state
        
        # Main training loop
        for it in pbar:
                
            self.opt_state = self.step(next(self.itercount), self.opt_state, batch)
                             
            if it % 100 == 0:

                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, batch)

                if (loss_value < loss_best) or jnp.isnan(loss_best) or jnp.isinf(loss_best): 
                    self.best_opt_state = self.opt_state
                    loss_best = loss_value

                # Store losses
                self.loss_training_log.append(loss_value)
                self.loss_best_log.append(loss_best)

                # 00000000000000000000000000000000000000000 losses
                pbar.set_postfix({'Loss': loss_value, 'Best': loss_best})
                
            batch= next(data) # Fetch data

        self.opt_state = self.best_opt_state

    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def predict(self, params, ubranch_star, ytrunk_star):
        
        cpred = 0
        nonlin_pred = self.nonlin_apply(params, ubranch_star, ytrunk_star)
        cpred += nonlin_pred
        return cpred
    


