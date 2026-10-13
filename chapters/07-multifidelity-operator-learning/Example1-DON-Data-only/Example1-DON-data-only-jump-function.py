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
# this will be used to construct the branch and trunk networks (see Snippet 1 in book chapter)
class MLP(eqx.Module):
    layers: list
    activation: jax.nn.identity

    def __init__(self, key, widths, activation):
        depth = len(widths) - 1
        keys = jax.random.split(key, depth)
        # Define a list of fully-connected layers
        self.layers = [eqx.nn.Linear(widths[i], widths[i+1], key=keys[i]) for i in range(depth)]
        # Define the activation function
        self.activation = activation
                      
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


# Deep operator network block, including a branch network and a trunk network
# Implements Equation (2)-(3) and Snippet 1 in the book chapter
class DeepONet(eqx.Module):
    branch: MLP
    trunk: MLP
    
    def __init__(self, key, branch_widths, branch_activation, trunk_widths, trunk_activation):
        branch_key, trunk_key = jax.random.split(key, 2)
        self.branch = MLP(branch_key, branch_widths, branch_activation)
        self.trunk = MLP(trunk_key, trunk_widths, trunk_activation)
    
    # The forward call involves an inner product between the output from trunk (basis functions)
    # and output from branch (coefficients): G(u)(y) = sum_k b_k(u) * t_k(y)
    # Input u: shape (m,), trunk coordinate y: shape (1,)
    # Output: scalar prediction
    def __call__(self, u, y):
        branch_output = self.branch(u)
        trunk_output = self.trunk(y)
        assert branch_output.shape == trunk_output.shape, "Outputs of branch and trunk must have the same dimensions"
        out = (branch_output * trunk_output).sum()
        return out
    
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

#%% Standard mean squared-error loss function (see Equation 4 & Snippet 2 in book chapter)

@eqx.filter_jit  # compile this function to make it run fast.
def loss(model, data):
    u, y, v = data
    pred_v = jax.vmap(model, in_axes=(0,0))(u,y)  # vectorise the model over a batch of data
    return optax.losses.l2_loss(pred_v, v).mean() # L2 loss

#%% Training algorithm

# Takes as input the untrained DeepONet model, optimizer and datasets and returns the trained model
def train(
    model: DeepONet, #untrained model (input)
    optim: optax.GradientTransformation, #optimizer
    train_data: list, #training dataset
    valid_data: list, #validation dataset
    batch_size: int,  #batch size
    key: Array,       #key for the random number generator for sampling batches
    steps: int,       #number of training steps
    print_every: int, #frequency of priniting training stats
    ) -> DeepONet:    #trained model (output)

    # It only makes sense to train the arrays in our model,
    # so we filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    # This defines a single optimization step:
    # given a model and a batch of data, compute loss, backpropagate to calculate gradients, then update model's weights
    def make_step( 
        model: DeepONet,
        opt_state: PyTree,
        data: list
    ):

        #differentiate all floating-point arrays in `model`.
        grads = eqx.filter_grad(loss)(model, data)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    train_loss = evaluate(model, train_data)
    valid_loss = evaluate(model, valid_data)
    print(
        f"\rtraining step={0}: train_loss={train_loss}, "
        f"validation_loss={valid_loss}"
    )

    train_loader_key, key = jax.random.split(key, 2)
    train_loader = dataloader(train_data, batch_size, key=train_loader_key)
    for step in range(1, steps+1):
        
        #sample one batch from the training dataset
        data = next(train_loader)
        
        #perfom a single optimization step
        model, opt_state = make_step(model, opt_state, data)
        
        #print training performance
        if (step % print_every) == 0 or (step == steps):
            train_loss = evaluate(model, train_data)
            valid_loss = evaluate(model, valid_data)
            print(
                f"\rtraining step={step}: train_loss={train_loss}, "
                f"validation_loss={valid_loss}"
            )
            
    return model

# A function to evaluate the model performance
def evaluate(model: DeepONet, data: list):
    """This function evaluates the model on the given dataset,
    computing the average loss and possibly any additional metrics.
    """
    # Note that all the JAX operations happen inside `loss`
    # 'loss' has JIT wrappers, so this is fast.
    avg_loss = loss(model, data)
    return avg_loss

#%% Reference solutions to train and test the model
def solution_fun(a, y):
    u =  a * y - 4
    if y <= 0.5:
        v = (6*y - 2)**2 * np.sin(u)
    else:
        v = (6*y - 2)**2 * np.sin(u) + 6
    return v

def input_fun(a, x):
    u = a * x - 4
    return u

#%% Setting the seed of the random number generator, for reproducibility
SEED = 125678
key = jax.random.PRNGKey(SEED)

#%% Problem Setup

x0, x1 = 0, 1   #domain boundaries
a0, a1 = 10, 14 #range of a values

# For Figure 2 results, m = 30, N = 15
# For Figure 6 results, m = 4, N = 10

m = 4 #number of sampling points for input function
x_vals = np.linspace(x0, x1, m) #sampling points for the branch net
y_vals = x_vals #locations/coordinates of output evaluations

N = 10   #number of input functions in the dataset
a_vals = np.linspace(a0, a1, N)

#%% DeepONet architecture 

branch_widths = [m, 30, 30, 30] #the branch takes the input function at m sample locations
trunk_widths  = [1, 30, 30, 30] #the trunk net takes a single coordinate (y)

branch_activation = jax.nn.tanh #activation function for the branch network
trunk_activation = jax.nn.tanh  #activation function for the trunk network


#%% Generate dataset

udata, ydata, vdata = [], [], [] #[u, y, v]
for a in a_vals:
    u = input_fun(a, x_vals) #sampling input functions at x_vals locations
    for y in y_vals:
        v = solution_fun(a, y)
    
        udata.append(u)
        ydata.append(y)
        vdata.append(v)
        
data = (jnp.array(udata), jnp.array(ydata).reshape([-1,1]), jnp.array(vdata))
dataset_size = len(udata)

#%% Create training and validation datasets

train_data_size = int(dataset_size*0.8)

key, split_key = jax.random.split(key, 2)
indices = jax.random.permutation(split_key, dataset_size)

train_data = tuple(data[i][indices[:train_data_size]] for i in range(3)) 
valid_data = tuple(data[i][indices[train_data_size:]] for i in range(3)) 

#%% Build model

key, model_key = jax.random.split(key, 2)
model =  DeepONet(model_key, branch_widths, branch_activation, trunk_widths, trunk_activation)

#%% Model training

# Quick verification mode support: run with --quick for a 100-step test
quick_mode = "--quick" in sys.argv
STEPS = 100 if quick_mode else 100_000 #number of training steps
LEARNING_RATE = 1e-3   #learning rate
BATCH_SIZE = 32        #batch size
PRINT_EVERY = 20 if quick_mode else 500 #frequency for printing training performance

if quick_mode:
    print(f"Running Example 1 in QUICK TEST mode (steps={STEPS}).")

## define optimizer
optim = optax.adamw(LEARNING_RATE) #Adam optimizer with weight decay regularization

key, loader_key = jax.random.split(key, 2)
model = train(model, optim,  
                     train_data, valid_data,
                     BATCH_SIZE, loader_key,
                     STEPS, PRINT_EVERY)

#%% Store model

outdir = './outputs/'
if not os.path.isdir(outdir):
    os.mkdir(outdir)

filename = outdir + 'model_don_m_' + str(m) + '_N_' + str(N) + '.eqx'
eqx.tree_serialise_leaves(filename, model)

#%% Test the model

atest = 12.0 #value of (a) for testing
ntest = 100  #number of evaluation points to plot the DeepONet prediction vs true solution
             #this shows that once trained, an operator network can be queried at any locations
ytest = np.linspace(x0, x1, ntest)


#the input function for the test case
#note that sampling *locations* should be the same as in training
utest = input_fun(atest, x_vals)

#true solution
vtest = np.zeros(ntest)
for i, y in enumerate(ytest):
    vtest[i] = solution_fun(atest, y)

#DeepONet prediction
vpred = jax.vmap(model, in_axes=(None, 0))(utest, ytest.reshape(-1,1))

#relative l2 error
err = jnp.linalg.norm(vtest-vpred)/jnp.linalg.norm(vtest)
print(f"Relative l2 error is {err}")


#high fidelity test data points (at the same locations in training, included for reference)
vtesthigh = np.zeros_like(y_vals)
for i, y in enumerate(y_vals):
    vtesthigh[i] = solution_fun(atest, y)

#%% Plotting results
fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(8,4))

ax.plot(ytest, vtest, label=r'\bf{Exact}', color = 'C0', linewidth=2)
ax.plot(ytest, vpred, label=r'\bf{DON Prediction}', linestyle='-', color = 'C3', linewidth=2)
ax.plot(y_vals, vtesthigh, label=r'\bf{Reference Data Points}', linestyle='None', marker='o', markersize=8, color = 'C2')
ax.set_xlabel(r'$y$', fontsize=18)
ax.set_ylabel(r'$v$', fontsize=18)

ax.legend(ncol=1, fontsize=12, handletextpad=0.5, columnspacing=1.5, handlelength=3)
fig.subplots_adjust(hspace=0.9, wspace=0.35)

plotdir = './plots/'
if not os.path.isdir(plotdir):
    os.mkdir(plotdir)

if (m==30) and (N==15):
    plt.savefig(plotdir+'Example1_DON_jump_function.png', dpi = 500, bbox_inches = 'tight')
    plt.savefig(plotdir+'Example1_DON_jump_function.pdf', dpi = 500, bbox_inches = 'tight')

if (m==4) and (N==10):
    plt.savefig(plotdir+'Example3_DON_jump_function.png', dpi = 500, bbox_inches = 'tight')
    plt.savefig(plotdir+'Example3_DON_jump_function.pdf', dpi = 500, bbox_inches = 'tight')