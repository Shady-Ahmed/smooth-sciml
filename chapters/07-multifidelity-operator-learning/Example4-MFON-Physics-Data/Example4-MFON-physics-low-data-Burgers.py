#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code accompanying the book chapter "Multifidelity Operator Learning" 
    Authors: Amanda A. Howard, Panos Stinis, and Shady E. Ahmed
    For comments or inquiries, please contact: shady.ahmed@pnnl.gov

This code is part of the chapter published in "An Introduction to Scientific
    Machine Learning: Mathematical Foundations and Numerical Implementation"
"""

#%% Loading libraries

# operating system main tools
import os
import sys
#os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

# linear algebra, optimization, and DL libraries
import jax
import optax
import equinox as eqx
import numpy as np
import jax.numpy as jnp

# data visualization tools
import matplotlib.pyplot as plt

# formatting tools
from jaxtyping import Array, Float, PyTree  # https://github.com/google/jaxtyping

#%% Extra platform settings
"""
Training a mulifidelity physics-informed DeepONet can be expensive and would benefit from GPU acceleration
Users can experiment with smaller network sizes, smaller datasets, or fewer training steps
"""
# Global flag to set a specific platform, must be used at startup.
# more details: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
jax.config.update('jax_platform_name', 'cpu') #change to 'gpu', if available
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
        'size'   : 14}
mpl.rc('font', **font)

#%% Neural network functionalities

# Multi-layer perceptron block (feedforward neural network)
# this will be used to construct the branch and trunk networks
class MLP(eqx.Module):
    layers: list
    activation: jax.nn.identity

    def __init__(self, key, widths, activation):
        depth = len(widths) - 1
        keys = jax.random.split(key, depth)
        #define a list of fully-connected layers
        self.layers = [eqx.nn.Linear(widths[i], widths[i+1], key=keys[i]) for i in range(depth)]
        #define the activation function
        self.activation = activation
                      
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

# Deep operator network block, including a branch network and a trunk network
class DeepONet(eqx.Module):
    branch: MLP
    trunk: MLP
    
    def __init__(self, key, branch_widths, branch_activation, trunk_widths, trunk_activation):
        branch_key, trunk_key = jax.random.split(key, 2)
        self.branch = MLP(branch_key, branch_widths, branch_activation)
        self.trunk = MLP(trunk_key, trunk_widths, trunk_activation)
                 
    # The forward call involves a dot product between the output from trunk (basis) and output from branch (coefficients)                  
    def __call__(self, u, y):
        branch_output = self.branch(u)
        trunk_output = self.trunk(y)
        assert branch_output.shape == trunk_output.shape, "Outputs of branch and trunk must have the same dimensions"
        out = (branch_output * trunk_output).sum()
        return out
    
    # wrapper function to calrify differentiation with respect to x or t
    def expand(self, u, x, t):
        y = jnp.array([x,t])
        return self(u, y)
    

# Multifidelity operator network block, including a low-fidelity DeepONet and two additional DeepONet
# representing the linear and nonlinear correlations between low-fidelity and high-fidelity predictions    
class MFON(eqx.Module):
    lf_subnet: DeepONet
    lin_subnet: DeepONet    
    nonlin_subnet: DeepONet
    xhigh: Float[Array, " Qh"]

    def __init__(self, key,
                 lf_branch_widths, lf_branch_activation, lf_trunk_widths, lf_trunk_activation,
                 lin_branch_widths, lin_trunk_widths,
                 nonlin_branch_widths, nonlin_branch_activation, nonlin_trunk_widths, nonlin_trunk_activation,
                 xhigh):
        lf_key, lin_key, nonlin_key = jax.random.split(key, 3)
        self.lf_subnet = DeepONet(lf_key, lf_branch_widths, lf_branch_activation, lf_trunk_widths, lf_trunk_activation)
        self.lin_subnet = DeepONet(lin_key, lin_branch_widths, jax.nn.identity, lin_trunk_widths, jax.nn.identity)
        self.nonlin_subnet = DeepONet(nonlin_key, nonlin_branch_widths, nonlin_branch_activation, nonlin_trunk_widths, nonlin_trunk_activation)
       
        # For MFON, the low-fidelity subnet is queried at xhigh locations
        # to expand the input samples to the nonlinear correlation subnetwork
        self.xhigh = xhigh.reshape(-1,1)
        
    def __call__(self, ulf, uhf, y):
        #first, the low-fidelity prediction at location y (in x-t grid) is computed
        lf_out = self.lf_subnet(ulf, y)
        
        #create y coordinates using the xhigh (space) and current t (time) 
        yhigh = jnp.hstack([self.xhigh, y[1]*jnp.ones_like(self.xhigh)])
        #second, the low-fidelity subnet is queried at yhigh locations to expand the input samples to the nonlinear correlation subnetwork
        lf_yhf = jax.vmap(self.lf_subnet, in_axes=(None, 0))(ulf, yhigh).reshape(-1,)

        #finally, both the linear and nonlinear blocks are called, and their outputs are summed
        lin_out = self.lin_subnet(jnp.array([lf_out]), y)
        nonlin_out = self.nonlin_subnet(jnp.hstack([uhf, lf_yhf]), y)
        out = lin_out + nonlin_out
        return out
    
    #low-fidelity block alone
    def low_fidelity(self, ulf, y):
        return self.lf_subnet(ulf, y)
    
    #high-fidelity block, which includes both low-fidelity predictions and high-fidelity correlations (i.e., multifidelity)
    def high_fidelity(self, ulf, uhf, y):
        lf_out = self.lf_subnet(ulf, y)
        yhf = jnp.hstack([self.xhigh, y[1]*jnp.ones_like(self.xhigh)])
        lf_yhf = jax.vmap(self.lf_subnet, in_axes=(None,0))(ulf, yhf).reshape(-1,)
        lin_out = self.lin_subnet(jnp.array([lf_out]), y)
        nonlin_out = self.nonlin_subnet(jnp.hstack([uhf, lf_yhf]), y)
        out = lin_out + nonlin_out
        return out
   
    # Wrapper function to clarify differentiation with respect to spatial coordinate x or time t
    def expand(self, ulf, uhf, x, t):
        y = jnp.array([x,t])
        return self(ulf, uhf, y)
    
    
#%% Simple data loader for sampling batches from the dataset

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
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
            
#%% Loss functions (Multifidelity physics-informed DeepONet: low-fidelity data + high-fidelity physics)

# Low-fidelity loss (pure data-driven MSE loss on coarse simulation data)
@eqx.filter_jit  # compile this function to make it run fast.
def loss_low(model, data_low):
    u, y, v = data_low
    pred_v = jax.vmap(model.low_fidelity, in_axes=(0,0))(u,y)  # vectorise the model over a batch of data
    return optax.losses.l2_loss(pred_v, v).mean() #L2 loss

# High-fidelity loss: physics-informed composite enforcing IC, periodic BC, and PDE residual
@eqx.filter_jit  # compile this function to make it run fast.
def loss_high(model, physics_high):
    ic_data, bc0_data, bc1_data, res_data = physics_high
    l_ic = IC_loss(model, ic_data)
    l_bc = BC_loss(model, bc0_data, bc1_data)
    l_res = PDE_loss(model, res_data)
    l1, l2, l3 = 20, 1, 10   
    l_tot = l1*l_ic + l2*l_bc + l3*l_res 
    return l_tot

# initial condition loss term
@eqx.filter_jit  # compile this function to make it run fast.
def IC_loss(model, data):
    ul, uh, y, v = data
    pred_v = jax.vmap(model, in_axes=(0,0,0))(ul, uh, y)  # vectorise the model over a batch of data
    return optax.losses.l2_loss(pred_v, v).mean()  # L2 loss

# boundary condition loss term
@eqx.filter_jit  # compile this function to make it run fast.
def BC_loss(model, data0, data1):
    ulf, uhf, y0, v = data0
    ulf, uhf, y1, v = data1
    
    #assert jnp.allclose(y0[:,1], y1[:,1]), "Mismatching times for BCs compuations"

    pred_v1 = jax.vmap(model, in_axes=(0,0,0))(ulf, uhf, y0)  # vectorise the model over a batch of data
    pred_v2 = jax.vmap(model, in_axes=(0,0,0))(ulf, uhf, y1)  # vectorise the model over a batch of data

    pred_v_dx1 = jax.vmap(jax.grad(model.expand, argnums=2), in_axes=(0,0,0,0))(ulf, uhf, y0[:,0], y0[:,1])
    pred_v_dx2 = jax.vmap(jax.grad(model.expand, argnums=2), in_axes=(0,0,0,0))(ulf, uhf, y1[:,0], y1[:,1])
   
    lb1 = optax.losses.l2_loss(pred_v1, pred_v2).mean()  # L2 loss
    lb2 = optax.losses.l2_loss(pred_v_dx1, pred_v_dx2).mean() # L2 loss
    return lb1 + lb2

# PDE residual loss term
@eqx.filter_jit  # compile this function to make it run fast.
def PDE_loss(model, data):
    ulf, uhf, y, v = data
   
    #compute v
    pred_v = jax.vmap(model, in_axes=(0,0,0))(ulf, uhf, y)  # vectorise the model over a batch of data
    #compute first derivative of v with respect to t
    pred_v_dt = jax.vmap(jax.grad(model.expand, argnums=3), in_axes=(0,0,0,0))(ulf, uhf, y[:,0], y[:,1])
    #compute first derivative of v with respect to x
    pred_v_dx = jax.vmap(jax.grad(model.expand, argnums=2), in_axes=(0,0,0,0))(ulf, uhf, y[:,0], y[:,1])
    #compute second derivative of v with respect to x
    pred_v_dxx = jax.vmap(jax.grad(jax.grad(model.expand, argnums=2), argnums=2), in_axes=(0,0,0,0))(ulf, uhf, y[:,0], y[:,1])
    
    res = pred_v_dt + pred_v * pred_v_dx - nu * pred_v_dxx
    return optax.losses.l2_loss(res, jnp.zeros_like(v)).mean() # L2 loss

# total loss, includes low-fidelity, high-fidelity, and other regularization terms
@eqx.filter_jit  # compile this function to make it run fast.
def loss(model, data_low, physics_high):
    l_low = loss_low(model, data_low)
    l_high = loss_high(model, physics_high)
    nonlin_weight = l2_reg(model.nonlin_subnet.branch)
    lf_weight = l2_reg(model.lf_subnet.branch)

    l3, l4 = 1e-6, 1e-6 #loss terms/weights for nonlinear and low-fideliy subnets (regularization)
    l_tot = l_high + l_low + l3*nonlin_weight + l4*lf_weight
    return l_tot

#total loss, without the regularization terms (for evaluation purposes)
@eqx.filter_jit  # compile this function to make it run fast.
def loss_eval(model, data_low, physics_high):
    l_low = loss_low(model, data_low)
    l_high = loss_high(model, physics_high)
    l_tot = l_high + l_low 
    return l_tot

#regularization on model's weights
@eqx.filter_jit  # compile this function to make it run fast.
def l2_reg(model):    
    params = eqx.filter(model, eqx.is_array)
    loss = sum(
                (w**2).sum()
                for w in jax.tree.leaves(params)
                )
    return loss

#%% Training algorithm

# Takes as input the untrained MFON model, optimizer and datasets and returns the trained model
def train(
    model: MFON, #untrained model (input)
    optim: optax.GradientTransformation, #optimizer
    train_data_low: list,     #low-fidelity training dataset
    train_physics_high: list, #high-fidelity training dataset (sample points for IC, BC, PDE)
    valid_data_low: list,     #low-fidelity validation dataset
    valid_physics_high: list, #high-fidelity validation dataset (sample points for IC, BC, PDE)
    batch_size_low: int,      #batch size for the low-fidelity dataset
    batch_size_high: int,     #batch size for the high-fidelity datase
    key: Array,               #key for the random number generator for sampling batches
    steps: int,               #number of training steps
    print_every: int,         #frequency of priniting training stats
    ) -> MFON: #trained model (output)

    # It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    # This defines a single optimization step:
    # given a model and a batch of low-fidelity data + a batch of high-fidelity data,
    # compute loss, backpropagate to calculate gradients, then update model's weights
    def make_step(
        model: DeepONet,
        opt_state: PyTree,
        data_low: list,
        physics_high: list,
    ):

        # differentiate all floating-point arrays in `model`.
        grads = eqx.filter_grad(loss)(model, data_low, physics_high)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state
    
    train_loader_low_key, train_loader_high_key = jax.random.split(key, 2)
    train_loader_low = dataloader(train_data_low, batch_size_low, key=train_loader_low_key)
    
    #we choose to sample more points for IC and PDE residuals than BC (optional)
    train_ic_data, train_bc0_data, train_bc1_data, train_res_data = train_physics_high
    train_ic_loader_key, train_bc_loader_key, train_res_loader_key = jax.random.split(train_loader_high_key, 3)
    train_ic_loader = dataloader(train_ic_data, batch_size_high*2, key=train_ic_loader_key)
    train_bc0_loader = dataloader(train_bc0_data, batch_size_high, key=train_bc_loader_key)
    train_bc1_loader = dataloader(train_bc1_data, batch_size_high, key=train_bc_loader_key)
    train_res_loader = dataloader(train_res_data, batch_size_high*10, key=train_res_loader_key)
      
    data_low = next(train_loader_low)
    physics_high = next(train_ic_loader), next(train_bc0_loader), next(train_bc1_loader), next(train_res_loader)
    
    #the training dataset is very large, so we choose to evaluate on a small batch for efficiency
    #but in principle, we can evaluate on the whole set too (if resources permit)
    train_loss = evaluate(model, data_low, physics_high)
    
    #here the validation dataset is pretty small, so we can evaluate on the whole set
    valid_loss = evaluate(model, valid_data_low, valid_physics_high)
        
    print(
        f"\rtraining step={0}: train_loss={train_loss}, "
        f"validation_loss={valid_loss}"
    )    
        
    for step in range(1, steps+1):

        #sample one batch from the low-fidelity training dataset
        data_low = next(train_loader_low)
        #sample one batch from the high-fidelity training dataset
        physics_high = next(train_ic_loader), next(train_bc0_loader), next(train_bc1_loader), next(train_res_loader)
        
        #perfom a single optimization step
        model, opt_state = make_step(model, opt_state, data_low, physics_high)
        
        #print training performance
        if (step % print_every) == 0 or (step == steps):
            
            train_loss = evaluate(model, data_low, physics_high)
            #here the validation dataset is pretty small, so we can evaluate on the whole set
            valid_loss = evaluate(model, valid_data_low, valid_physics_high)
            print(
                f"\rtraining step={step}: train_loss={train_loss}, "
                f"validation_loss={valid_loss}"
            )
            
    return model


def evaluate(model: DeepONet, data_low: list, physics_high: list):
    """This function evaluates the model on the test dataset,
    computing the average loss and any arbitrary metrics.
    """
    # Note that all the JAX operations happen inside `loss`
    # 'loss' has JIT wrappers, so this is fast.
    avg_loss = loss_eval(model, data_low, physics_high)
    return avg_loss
       
        
#%% Preparing datasets
def prepare_deeponet_data_low(x, t, u0_batch, u_batch, target=True):
    """
    Converts batch of initial conditions and their solutions
    into DeepONet (branch, trunk) → output format.
    """
    N, nt, nx = u_batch.shape  # N samples, time steps, spatial points
        
    # Repeat each u0 for each pair (x,t)
    branch_inputs = np.repeat(u0_batch, repeats=nx*nt, axis=0)   # shape: (N * nx * nt, mlow)       
    
    # Create trunk input grid (x, t) of shape (nt * nx, 2)
    tt, xx = np.meshgrid(t, x, indexing='ij')
    xt_grid = np.hstack([xx.reshape(-1,1), tt.reshape(-1,1)])    
    trunk_inputs = np.tile(xt_grid, (N, 1))  # shape: (N * nt * nx, 2)

    targets = u_batch.reshape(N * nt * nx, )  # shape: (N * nt * nx,)

    return jnp.array(branch_inputs), jnp.array(trunk_inputs), jnp.array(targets) #[u, y, v]

def prepare_deeponet_data_high(x, t, u0_batch_low, u0_batch_high, u_batch, target=True):
    """
    Converts batch of initial conditions and their solutions
    into DeepONet (branch, trunk) → output format.
    """
    N, nt, nx = u_batch.shape  # N samples, time steps, spatial points

    # Create trunk input grid (x, t) of shape (nt * nx, 2)
    if (nt==1) or (nx==1): #IC or BC
    
        # Repeat each u0 for each point in x and t
        branch_inputs_low = np.repeat(u0_batch_low, repeats=nt*nx, axis=0)  # shape: (N * nt * nx, mlow)
        branch_inputs_high = np.repeat(u0_batch_high, repeats=nt*nx, axis=0)  # shape: (N * nt * nx, mhigh)
    
        xt_grid = np.array([[xi, ti] for ti in t for xi in x])
           
    else:
        assert x.shape == t.shape, "Cannot figure out x and t grids, please check dimensions"

        # Repeat each u0 for each pair (x,t)
        branch_inputs_low = np.repeat(u0_batch_low, repeats=x.shape[0], axis=0)   # shape: (N * nx, mlow)     
        branch_inputs_high = np.repeat(u0_batch_high, repeats=x.shape[0], axis=0)   # shape: (N * nx, mhigh)     

        xt_grid = np.array([[xi, ti] for (xi,ti) in zip(x,t)])

    trunk_inputs = np.tile(xt_grid, (N, 1))  # shape: (N * nt * nx, 2)

    if target:
        # Target outputs: u(x, t) flattened for each trajectory
        targets = u_batch.reshape(N * nt * nx)  # shape: (N * nt * nx,)
    else:
        targets = np.stack([0.0]*trunk_inputs.shape[0]) #just dummy value -- not used
    return jnp.array(branch_inputs_low), jnp.array(branch_inputs_high), jnp.array(trunk_inputs), jnp.array(targets) #[u, y, v]


# utility function to split datasets into training and validation sets
def train_valid_split(key, dataset, train_fraction):
    
    assert dataset[0].shape[0] == dataset[1].shape[0] == dataset[2].shape[0], "Inconsistent sizes of data arrays"
    
    dataset_size = dataset[0].shape[0]
    indices = jax.random.permutation(key, dataset_size)
    train_data_size = int(dataset_size*train_fraction)

    train_data = tuple(dataset[i][indices[:train_data_size]] for i in range(len(dataset))) 
    valid_data = tuple(dataset[i][indices[train_data_size:]] for i in range(len(dataset))) 

    return train_data, valid_data

#%% Setting the seed of the random number generator, for reproducibility
SEED = 125678
key = jax.random.PRNGKey(SEED)

#%% Problem Setup

nu = 1e-2 #kinematic viscosity

x0, x1 = 0.0, 1.0 #spatial domain boundaries
t0, t1 = 0.0, 1.0 #time domain

mlow = 9 #number of low fidelity initial condition sensors (for each input function)
x_ic_low = np.linspace(x0, x1, mlow) #locations of the low fidelity initial condition sensors (uniformly spaced)
xlow = x_ic_low #locations/coordinates of low fidelity output evaluations

mhigh = 257 #number of high fidelity initial condition sensors (for each input function)
x_ic_high = np.linspace(x0, x1, mhigh) #locations of the high fidelity initial condition sensors (uniformly spaced)
xhigh = x_ic_high #locations/coordinates of high fidelity output evaluations

nt_bc_high = 101 #number of boundary condition samples (for each input function)
key, bc_key = jax.random.split(key, 2)
t_bc_high = jax.random.uniform(bc_key, shape=(nt_bc_high,), minval=t0, maxval=t1) #time instants for the boundary condition samples (randmoly sampled)

n_res_high = 2500 #number of sampling points for residual calculation
key, res_keyx, res_keyt = jax.random.split(key, 3)
x_res_high = jax.random.uniform(res_keyx, shape=(n_res_high,), minval=x0, maxval=x1) #locations for the residual calculation samples (randmoly sampled)
t_res_high = jax.random.uniform(res_keyt, shape=(n_res_high,), minval=t0, maxval=t1) #time instants for the residual calculation samples (randmoly sampled)

nt_low = 101  #number of time instants of available low fidelity data between 0 and 1
tlow = np.linspace(t0, t1, nt_low)

nt_high = 101 #number of time instants of available high fidelity data between 0 and 1
thigh = np.linspace(t0, t1, nt_high)

#%% MFON  architecture

lf_branch_widths = [mlow, *5*[64]]  #for the low-fidelity networ, the branch takes the input function at mlow sample locations
lf_trunk_widths  = [2, *5*[64]]     #the trunk net takes x-t coordinates (y)

lin_branch_widths = [1, 64]  #for linear correlation subnet,  branch takes the low-fidelity prediction as input
lin_trunk_widths  = [2, 64]  #the trunk net takes x-t coordinates (y)

nonlin_branch_widths = [2*mhigh, *5*[64]]  #for nonlinear correlation subnet,  branch takes the input function
                                           #at migh sample locations in addition to the low-fidelity predictions at these locations 
nonlin_trunk_widths  = [2, *5*[64]]        #the trunk net takes x-t coordinates (y)

#activation functions
lf_branch_activation = jax.nn.tanh
lf_trunk_activation = jax.nn.tanh
nonlin_branch_activation = jax.nn.tanh
nonlin_trunk_activation = jax.nn.tanh

#%% Read reference solutions and construct training and validation sets

Ntot_low = 10 #number of input functions in the dataset
Nlow = 8  #number of input functions in the low fidelity training/validation dataset

Ntot_high = 10 #number of input functions in the dataset
Nhigh = 8 #number of input functions in the high fidelity training/validation dataset

#------------------------- Low Fidelity Dataset ------------------------------#
filename_low = './burgers-data/Example4_burgers_periodic_' + str(Ntot_low) + '_ICS_low.npy'
if not os.path.isfile(filename_low):
    raise FileNotFoundError(
        f"Low-fidelity reference file '{filename_low}' not found. "
        "Please run 'Example4-Burgers-reference-solutions-high-and-low.py' first."
    )
usol_low = np.load(filename_low)

#----low fidelity dataset----#
lf_data = prepare_deeponet_data_low(xlow, tlow, usol_low[:Nlow, 0, :], usol_low[:Nlow, :, :])


#------------------------ High Fidelity Dataset ------------------------------#
filename_high = './burgers-data/Example4_burgers_periodic_' + str(Ntot_high) + '_ICS_high.npy'
if not os.path.isfile(filename_high):
    raise FileNotFoundError(
        f"High-fidelity reference file '{filename_high}' not found. "
        "Please run 'Example4-Burgers-reference-solutions-high-and-low.py' first."
    )
usol_high = np.load(filename_high)
stride = int((mhigh - 1)/(mlow - 1))

#----IC dataset----#
ic_data = prepare_deeponet_data_high(x_ic_high, [t0], usol_high[:Nhigh, 0, ::stride], usol_high[:Nhigh, 0, :], usol_high[:Nhigh, [0], :])

#----BC dataset----#
bc0_data = prepare_deeponet_data_high([x0], t_bc_high, usol_high[:Nhigh, 0, ::stride], usol_high[:Nhigh ,0, :], usol_high[:Nhigh ,: ,[0]], target=False)
bc1_data = prepare_deeponet_data_high([x1], t_bc_high, usol_high[:Nhigh, 0, ::stride], usol_high[:Nhigh ,0, :], usol_high[:Nhigh ,: ,[0]], target=False)

#----residual dataset----#
res_data = prepare_deeponet_data_high(x_res_high, t_res_high, usol_high[:Nhigh, 0, ::stride], usol_high[:Nhigh, 0, :], usol_high[:Nhigh, :, :], target=False)

#%% Create train/valid datasets

# low fidelity dataset [data]
key, data_key = jax.random.split(key, 2)
train_data_low, valid_data_low = train_valid_split(data_key, lf_data, train_fraction=0.8)

# high fidelity dataset [physics]
key, split_key_ic, split_key_bc, split_key_res = jax.random.split(key, 4)
train_ic_data, valid_ic_data = train_valid_split(split_key_ic, ic_data, train_fraction=0.8)
train_bc0_data, valid_bc0_data = train_valid_split(split_key_bc, bc0_data, train_fraction=0.8)
train_bc1_data, valid_bc1_data = train_valid_split(split_key_bc, bc1_data, train_fraction=0.8)
train_res_data, valid_res_data = train_valid_split(split_key_res, res_data, train_fraction=0.8)

train_physics_high = train_ic_data, train_bc0_data, train_bc1_data, train_res_data
valid_physics_high = valid_ic_data, valid_bc0_data, valid_bc1_data, valid_res_data
   
#%% Build model

key, model_key = jax.random.split(key, 2)
model =  MFON(model_key,
              lf_branch_widths, lf_branch_activation, lf_trunk_widths, lf_trunk_activation,
              lin_branch_widths, lin_trunk_widths,
              nonlin_branch_widths, nonlin_branch_activation, nonlin_trunk_widths, nonlin_trunk_activation, jnp.array(x_ic_high))

#%% Model training

quick_mode = "--quick" in sys.argv
train_flag = "--train" in sys.argv

EPOCHS = 100 if quick_mode else 10_000   #number of training steps
LEARNING_RATE = 1e-3    #learning rate
batch_size_high = 512   #batch size for the low fidelity dataset
batch_size_low = 4096   #batch size for the high fidelity dataset
PRINT_EVERY = 20 if quick_mode else 500  #frequency for printing training performance

Train = train_flag or quick_mode

outdir = './outputs/'
if not os.path.isdir(outdir):
    os.mkdir(outdir)
filename = outdir + 'Example4_model_mfon_physics_mhigh_' + str(mhigh) + '_Nhigh_' + str(Nhigh) \
                  + '_mlow_' + str(mlow) + '_Nlow_' + str(Nlow) + 'epochs_' + str(EPOCHS) + '.eqx'

if quick_mode:
    print(f"Running Example 4 in QUICK TEST mode (steps={EPOCHS}).")
                         
if Train:
    ## define optimizer
    learning_rate = LEARNING_RATE
    optim = optax.adamw(learning_rate)
    
    key, loader_key = jax.random.split(key, 2)
    
    # train model
    model = train(model, optim, 
                           train_data_low, train_physics_high, 
                           valid_data_low, valid_physics_high,
                           batch_size_low, batch_size_high,
                           loader_key, EPOCHS, PRINT_EVERY)
    # store model
    eqx.tree_serialise_leaves(filename, model)

else:
    #load model
    model = eqx.tree_deserialise_leaves(filename, model)

#%% Test the model

itest = -1  #test at the last initial condition (it was not part of the training set)
utest_low = usol_high[itest,0,::stride]
utest_high = usol_high[itest,0,:]

#true solution
vtest_high = usol_high[itest,:,:]


# define some x-t coordinates to query the trained MFON model
n_t, n_x = vtest_high.shape
xtest = np.linspace(x0, x1, n_x)
ttest = np.linspace(t0, t1, n_t)
tt, xx = np.meshgrid(ttest, xtest, indexing='ij')
ytest = np.hstack([xx.reshape(-1,1), tt.reshape(-1,1)])

#MFON prediction
vpred = jax.vmap(model, in_axes=(None, None, 0))(utest_low, utest_high, ytest)
vpred = vpred.reshape([n_t, n_x])

#relative l2 error
err = jnp.linalg.norm(vtest_high-vpred)/jnp.linalg.norm(vtest_high)
print(f"Relative l2 error is {err}")


#%% Plotting results

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,4))

labels = [r'\bf{True}', r'\bf{PI-MFON}']
levels = np.linspace(vtest_high.min(), vtest_high.max(), 50)
ax[0].contourf(xtest, ttest, vtest_high, levels=levels, cmap='turbo')
ax[1].contourf(xtest, ttest, vpred, levels=levels, cmap='turbo')

for i in range(2):
    ax[i].set_title(labels[i])
    ax[i].set_xlabel(r'$x$', fontsize=18)
    ax[i].set_ylabel(r'$t$', fontsize=18)

for ii, idx in enumerate([0, 50, 100]):
    if idx == 0:
        labels = [r'\bf{True}', r'\bf{PI-MFON}']
    else:
        labels = [None, None]
        
    ax[2].plot(xtest, vtest_high[idx], label=labels[0], color = 'C'+str(ii), linewidth=2, alpha=1)
    ax[2].plot(xtest, vpred[idx], label=labels[1], linestyle='--', color = 'C'+str(5), linewidth=2, alpha=1)

ax[2].set_xlabel(r'$x$', fontsize=18)
ax[2].set_ylabel(r'$v(x,t)$', fontsize=18)
ax[2].legend(ncol=1, fontsize=12, handletextpad=0.5, columnspacing=1.5, handlelength=3)


fig.subplots_adjust(hspace=0.9, wspace=0.35)

plotdir = './plots/'
if not os.path.isdir(plotdir):
    os.mkdir(plotdir)

plt.savefig(plotdir+'Example4_MFON_Burgers_physics_low_data.png', dpi = 500, bbox_inches = 'tight')
plt.savefig(plotdir+'Example4_MFON_Burgers_physics_low_data.pdf', dpi = 500, bbox_inches = 'tight')
