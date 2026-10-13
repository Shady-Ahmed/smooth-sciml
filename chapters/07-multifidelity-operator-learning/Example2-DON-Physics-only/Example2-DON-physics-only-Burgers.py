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

# linear algebra, optimization, and DL libraries
import jax
import optax
import equinox as eqx
import numpy as np
import jax.numpy as jnp

# data visualization tools
import matplotlib.pyplot as plt

# formatting tools
from jaxtyping import Array, PyTree  # https://github.com/google/jaxtyping

#%% Extra platform settings

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
    
    # The forward call involves an inner product between the output from trunk (basis)
    # and output from branch (coefficients): G(u)(y) = sum_k b_k(u) * t_k(y)
    # Input u: initial condition vector (m,), trunk y: spatial-temporal coordinates (x, t)
    def __call__(self, u, y):
        branch_output = self.branch(u)
        trunk_output = self.trunk(y)
        assert branch_output.shape == trunk_output.shape, "Outputs of branch and trunk must have the same dimensions"
        out = (branch_output * trunk_output).sum()
        return out
    
    # Wrapper function to clarify automatic differentiation with respect to spatial coordinate x or time t
    # Enables jax.grad(model.expand, argnums=1) for d/dx and argnums=2 for d/dt
    def expand(self, u, x, t):
        y = jnp.array([x,t])
        return self(u, y)
    
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

#%% Loss functions (Physics-informed DeepONet)

# Total training loss: composite loss balancing IC, periodic BC, and PDE residual
@eqx.filter_jit  # compile this function to make it run fast.
def loss(model, ic_data, bc0_data, bc1_data, res_data):
    l_ic = IC_loss(model, ic_data)
    l_bc = BC_loss(model, bc0_data, bc1_data)
    l_res = PDE_loss(model, res_data)
    
    # Weights for loss terms: lambda_ic, lambda_bc, lambda_res
    l1, l2, l3 = 20, 1, 10   

    l_tot = l1*l_ic + l2*l_bc + l3*l_res 
    return l_tot

# initial condition loss term
@eqx.filter_jit  # compile this function to make it run fast.
def IC_loss(model, data):
    u, y, v = data
    pred_v = jax.vmap(model, in_axes=(0, 0))(u,y)  # vectorise the model over a batch of data
    return optax.losses.l2_loss(pred_v, v).mean()  # L2 loss


# boundary condition loss term
@eqx.filter_jit  # compile this function to make it run fast.
def BC_loss(model, data0, data1):
    u, y0, v = data0
    u, y1, v = data1
    
    #assert jnp.allclose(y0[:,1], y1[:,1]), "Mismatching times for BCs compuations"

    pred_v1 = jax.vmap(model, in_axes=(0,0))(u, y0)  # vectorise the model over a batch of data
    pred_v2 = jax.vmap(model, in_axes=(0,0))(u, y1)  # vectorise the model over a batch of data

    pred_v_dx1 = jax.vmap(jax.grad(model.expand, argnums=1), in_axes=(0,0,0))(u, y0[:,0], y0[:,1])
    pred_v_dx2 = jax.vmap(jax.grad(model.expand, argnums=1), in_axes=(0,0,0))(u, y1[:,0], y1[:,1])
   
    lb1 = optax.losses.l2_loss(pred_v1, pred_v2).mean()  # L2 loss
    lb2 = optax.losses.l2_loss(pred_v_dx1, pred_v_dx2).mean() # L2 loss
    return lb1 + lb2

# PDE residual loss term
@eqx.filter_jit  # compile this function to make it run fast.
def PDE_loss(model, data):
    u, y, v = data
    
    #compute v
    pred_v = jax.vmap(model, in_axes=(0,0))(u, y)  # vectorise the model over a batch of data
    #compute first derivative of v with respect to t
    pred_v_dt = jax.vmap(jax.grad(model.expand, argnums=2), in_axes=(0,0,0))(u, y[:,0], y[:,1])
    #compute first derivative of v with respect to x
    pred_v_dx = jax.vmap(jax.grad(model.expand, argnums=1), in_axes=(0,0,0))(u, y[:,0], y[:,1])
    #compute second derivative of v with respect to x
    pred_v_dxx = jax.vmap(jax.grad(jax.grad(model.expand, argnums=1), argnums=1), in_axes=(0,0,0))(u, y[:,0], y[:,1])
    
    res = pred_v_dt + pred_v * pred_v_dx - nu * pred_v_dxx
    return optax.losses.l2_loss(res, jnp.zeros_like(v)).mean() # L2 loss

#%% Training algorithm

# Takes as input the untrained DeepONet model, optimizer and datasets and returns the trained model
def train(
    model: DeepONet, #untrained model (input)
    optim: optax.GradientTransformation, #optimizer
    train_ic_data: tuple,  #training samples for initial condition
    train_bc0_data: tuple, #training samples for first (left) boundary condition
    train_bc1_data: tuple, #training samples for second (right) boundary condition
    train_res_data: tuple, #training samples for PDE residual
    valid_ic_data: tuple,  #validation samples for initial condition
    valid_bc0_data: tuple, #validation samples for first (left) boundary condition
    valid_bc1_data: tuple, #validation samples for second (right) boundary condition
    valid_res_data: tuple, #validation samples for PDE residual
    batch_size: int,       #batch size
    key: Array,            #key for the random number generator for sampling batches
    steps: int,            #number of training steps
    print_every: int,      #frequency of priniting training stats
    ) -> DeepONet:

    # It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: DeepONet,
        opt_state: PyTree,
        ic_data: tuple,
        bc0_data: tuple,
        bc1_data: tuple,
        res_data: tuple
    ):

        # differentiate all floating-point arrays in `model`.
        grads = eqx.filter_grad(loss)(model, ic_data, bc0_data, bc1_data, res_data)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state

    train_loss = evaluate(model, train_ic_data, train_bc0_data, train_bc1_data, train_res_data)
    valid_loss = evaluate(model, valid_ic_data, valid_bc0_data, valid_bc1_data, valid_res_data)
    print(
        f"\rtraining step={0}: train_loss={train_loss}, "
        f"validation_loss={valid_loss}"
    )

    #we choose to sample more points for IC and PDE residuals than BC (optional)
    key, train_ic_loader_key, train_bc_loader_key, train_res_loader_key = jax.random.split(key, 4)
    train_ic_loader = dataloader(train_ic_data, batch_size*2, key=train_ic_loader_key)
    train_bc0_loader = dataloader(train_bc0_data, batch_size, key=train_bc_loader_key)
    train_bc1_loader = dataloader(train_bc1_data, batch_size, key=train_bc_loader_key)
    train_res_loader = dataloader(train_res_data, batch_size*10, key=train_res_loader_key)

    for step in range(1, steps+1):

        #sample one batch from the training datasets for IC, BCs, and PDE
        data_i = next(train_ic_loader)
        data_b0 = next(train_bc0_loader)
        data_b1 = next(train_bc1_loader)
        data_r = next(train_res_loader)

        assert jnp.allclose(data_b0[0], data_b1[0]), "Mismatching initial condition for BCs compuations"
        assert jnp.allclose(data_b0[1][:,1], data_b1[1][:,1]), "Mismatching times for BCs compuations"

        #perfom a single optimization step
        model, opt_state = make_step(model, opt_state, data_i, data_b0, data_b1, data_r)
        
        #print training performance
        if (step % print_every) == 0 or (step == steps):
            train_loss = evaluate(model, train_ic_data, train_bc0_data, train_bc1_data, train_res_data)
            valid_loss = evaluate(model, valid_ic_data, valid_bc0_data, valid_bc1_data, valid_res_data)
            print(
                f"\rtraining step={step}: train_loss={train_loss}, "
                f"validation_loss={valid_loss}"
            )
            
    return model


# A function to evaluate the model performance
def evaluate(model: DeepONet, ic_data: tuple, bc0_data: tuple, bc1_data: tuple, res_data: tuple):
    """This function evaluates the model on the given dataset,
    computing the average loss and possibly any additional metrics.
    """
    # Note that all the JAX operations happen inside `loss`
    # 'loss' has JIT wrappers, so this is fast.
    avg_loss = loss(model, ic_data, bc0_data, bc1_data, res_data)
    return avg_loss

#%% Preparing datasets
def prepare_deeponet_data(x, t, u0_batch, u_batch, target=True):
    """
    Converts batch of initial conditions and their solutions
    into DeepONet (branch, trunk) → output format.
    """
    N, nt, nx = u_batch.shape  # N samples, time steps, spatial points

    # Create trunk input grid (x, t) of shape (nt * nx, 2)
    if (nt==1) or (nx==1): #IC or BC
    
        # Repeat each u0 for each point in x and t
        branch_inputs = np.repeat(u0_batch, repeats=nt, axis=0)  # shape: (N * nt, n_ic)
        branch_inputs = np.repeat(branch_inputs, repeats=nx, axis=0)        # shape: (N * nt * nx, n_ic)
    
        xt_grid = np.array([[xi, ti] for ti in t for xi in x])
           
    else:
        assert x.shape == t.shape, "Cannot figure out x and t grids, please check dimensions"

        # Repeat each u0 for each pair (x,t)
        branch_inputs = np.repeat(u0_batch, repeats=x.shape, axis=0)   # shape: (N * nx, n_ic) = (N * nx, n_ic)       
        
        xt_grid = np.array([[xi, ti] for (xi,ti) in zip(x,t)])

    trunk_inputs = np.tile(xt_grid, (N, 1))  # shape: (N * nt * nx, 2)

    if target:
        # Target outputs: u(x, t) flattened for each trajectory
        targets = u_batch.reshape(N * nt * nx)  # shape: (N * nt * nx,)
    else:
        targets = np.stack([0]*trunk_inputs.shape[0]) #just dummy value -- not used
    return jnp.array(branch_inputs), jnp.array(trunk_inputs), jnp.array(targets) #[u, y, v]


# utility function to split datasets into training and validation sets
def train_valid_split(key, dataset, train_fraction):
    
    assert dataset[0].shape[0] == dataset[1].shape[0] == dataset[2].shape[0], "Inconsistent sizes of data arrays"
    
    dataset_size = dataset[0].shape[0]
    indices = jax.random.permutation(key, dataset_size)
    train_data_size = int(dataset_size*train_fraction)

    train_data = tuple(dataset[i][indices[:train_data_size]] for i in range(3)) 
    valid_data = tuple(dataset[i][indices[train_data_size:]] for i in range(3)) 

    return train_data, valid_data

#%% Setting the seed of the random number generator, for reproducibility
SEED = 125678
key = jax.random.PRNGKey(SEED)

#%% Problem Setup

nu = 1e-2 #kinematic viscosity

x0, x1 = 0.0, 1.0 #spatial domain boundaries
t0, t1 = 0.0, 1.0 #time domain

m = 257 #number of initial condition sensors (for each input function)
x_ic = np.linspace(x0, x1, m) #locations of the initial condition sensors (uniformly spaced)
y_vals = x_ic #locations/coordinates of output evaluations - can be arbitrary locations

nt_bc = 101 #number of boundary condition samples (for each input function)
key, bc_key = jax.random.split(key, 2)
t_bc = jax.random.uniform(bc_key, shape=(nt_bc,), minval=t0, maxval=t1) #time instants for the boundary condition samples (randmoly sampled)

n_res = 2500 #number of sampling points for residual calculation
key, res_keyx, res_keyt = jax.random.split(key, 3)
x_res = jax.random.uniform(res_keyx, shape=(n_res,), minval=x0, maxval=x1) #locations for the residual calculation samples (randmoly sampled)
t_res = jax.random.uniform(res_keyt, shape=(n_res,), minval=t0, maxval=t1) #time instants for the residual calculation samples (randmoly sampled)

# Load data
Ntot = 10   #number of input functions in the dataset
N = 8   #number of input functions in the training/validation dataset


#%% DeepONet architecture 
branch_widths = [m, *5*[64]] #the branch takes the input function at m sample locations
trunk_widths  = [2, *5*[64]] #the trunk net takes a single coordinate (y)

branch_activation = jax.nn.tanh #activation function for the branch network
trunk_activation = jax.nn.tanh  #activation function for the trunk network

#%% Read reference solutions and construct training and validation sets

filename = './burgers-data/Example2_burgers_periodic_' + str(Ntot) + '_ICS_high.npy'
if not os.path.isfile(filename):
    raise FileNotFoundError(
        f"Reference solution file '{filename}' not found. "
        "Please run 'Example2-Burgers-reference-solutions.py' first to generate it."
    )
usol = np.load(filename)

#----IC dataset----#
ic_data = prepare_deeponet_data(x_ic, [t0], usol[:N, 0, :], usol[:N, [0], :])

#----BC dataset----#
bc0_data = prepare_deeponet_data([x0], t_bc, usol[:N ,0, :], usol[:N ,: ,[0]], target=False)
bc1_data = prepare_deeponet_data([x1], t_bc, usol[:N ,0, :], usol[:N ,: ,[0]], target=False)

#----residual dataset----#
res_data = prepare_deeponet_data(x_res, t_res, usol[:N, 0, :], usol[:N, :, :], target=False)

# Create train/valid datasets
key, split_key_ic, split_key_bc, split_key_res = jax.random.split(key, 4)
train_ic_data, valid_ic_data = train_valid_split(split_key_ic, ic_data, train_fraction=0.8)
train_bc0_data, valid_bc0_data = train_valid_split(split_key_bc, bc0_data, train_fraction=0.8)
train_bc1_data, valid_bc1_data = train_valid_split(split_key_bc, bc1_data, train_fraction=0.8)
train_res_data, valid_res_data = train_valid_split(split_key_res, res_data, train_fraction=0.8)

#%% Build model

key, model_key = jax.random.split(key, 2)
model =  DeepONet(model_key, branch_widths, branch_activation, trunk_widths, trunk_activation)

#%% Model training

quick_mode = "--quick" in sys.argv
train_flag = "--train" in sys.argv

STEPS = 100 if quick_mode else 100_000 #number of training steps
LEARNING_RATE = 1e-3   #learning rate
BATCH_SIZE = 512       #batch size
PRINT_EVERY = 20 if quick_mode else 500 #frequency for printing training performance

Train = train_flag or quick_mode

outdir = './outputs/'
if not os.path.isdir(outdir):
    os.mkdir(outdir)
filename = outdir + 'Example2_model_don_physics_m_' + str(m) + '_N_' + str(N) + '_epochs' + str(STEPS) + '.eqx'

if quick_mode:
    print(f"Running Example 2 in QUICK TEST mode (steps={STEPS}).")

if Train:
    ## define optimizer
    learning_rate = LEARNING_RATE
    optim = optax.adamw(learning_rate)
    
    key, loader_key = jax.random.split(key, 2)
    
    # train model
    model = train(model, optim,  
                           train_ic_data, train_bc0_data, train_bc1_data, train_res_data,
                           valid_ic_data, valid_bc0_data, valid_bc1_data, valid_res_data,
                           BATCH_SIZE, loader_key,
                           STEPS, PRINT_EVERY)
    
    # store model
    eqx.tree_serialise_leaves(filename, model)

else:
    #load model
    model = eqx.tree_deserialise_leaves(filename, model)


#%% Test the model

itest = -1 #test at the last initial condition (it was not part of the training set)
utest = usol[itest,0,:]

#true solution
vtest = usol[itest,:,:]


# define some x-t coordinates to query the trained DeepONet model
n_t, n_x = vtest.shape
xtest = np.linspace(x0, x1, n_x)
ttest = np.linspace(t0, t1, n_t)

tt, xx = np.meshgrid(ttest, xtest, indexing='ij')
ytest = np.hstack([xx.reshape(-1,1), tt.reshape(-1,1)])

#DeepONet prediction
vpred = jax.vmap(model, in_axes=(None, 0))(utest, ytest)
vpred = vpred.reshape([n_t, n_x])

#relative l2 error
err = jnp.linalg.norm(vtest-vpred)/jnp.linalg.norm(vtest)
print(f"Relative l2 error is {err}")


#%% Plotting results

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,4))

labels = [r'\bf{True}', r'\bf{PI-DeepONet}']
levels = np.linspace(vtest.min(), vtest.max(), 50)
ax[0].contourf(xtest, ttest, vtest, levels=levels, cmap='turbo')
ax[1].contourf(xtest, ttest, vpred, levels=levels, cmap='turbo')

for i in range(2):
    ax[i].set_title(labels[i])
    ax[i].set_xlabel(r'$x$', fontsize=18)
    ax[i].set_ylabel(r'$t$', fontsize=18)

for ii, idx in enumerate([0, 50, 100]):
    if idx == 0:
        labels = [r'\bf{True}', r'\bf{PI-DeepONet}']
    else:
        labels = [None, None]
        
    ax[2].plot(xtest, vtest[idx], label=labels[0], color = 'C'+str(ii), linewidth=2, alpha=1)
    ax[2].plot(xtest, vpred[idx], label=labels[1], linestyle='--', color = 'C'+str(5), linewidth=2, alpha=1)

ax[2].set_xlabel(r'$x$', fontsize=18)
ax[2].set_ylabel(r'$v(x,t)$', fontsize=18)
ax[2].legend(ncol=1, fontsize=12, handletextpad=0.5, columnspacing=1.5, handlelength=3)


fig.subplots_adjust(hspace=0.9, wspace=0.35)

plotdir = './plots/'
if not os.path.isdir(plotdir):
    os.mkdir(plotdir)

plt.savefig(plotdir+'Example2_DON_Burgers_physics_only.png', dpi = 500, bbox_inches = 'tight')
plt.savefig(plotdir+'Example2_DON_Burgers_physics_only.pdf', dpi = 500, bbox_inches = 'tight')
