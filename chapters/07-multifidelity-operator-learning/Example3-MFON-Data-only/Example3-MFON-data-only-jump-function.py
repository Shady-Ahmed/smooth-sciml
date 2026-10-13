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
from jaxtyping import Array, Float, PyTree  # https://github.com/google/jaxtyping

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
    
    # The forward call involves a dot product between the output from trunk (basis) and output from branch (coefficients)                  
    def __call__(self, u, y):
        branch_output = self.branch(u)
        trunk_output = self.trunk(y)
        assert branch_output.shape == trunk_output.shape, "Outputs of branch and trunk must have the same dimensions"
        out = (branch_output * trunk_output).sum()
        return out #jnp.array([out])


# Multifidelity operator network block, including a low-fidelity DeepONet and two additional DeepONets
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
        #first, the low-fidelity prediction at location y is computed
        lf_out = self.lf_subnet(ulf, y).reshape(-1,)
        
        #second, the low-fidelity subnet is queried at xhigh locations to expand the input samples to the nonlinear correlation subnetwork
        lf_xhf = jax.vmap(self.lf_subnet, in_axes=(None,0))(ulf, self.xhigh).reshape(-1,)

        #finally, both the linear and nonlinear blocks are called, and their outputs are summed
        lin_out = self.lin_subnet(lf_out,y)
        nonlin_out = self.nonlin_subnet(jnp.hstack([uhf, lf_xhf]), y)
        out = lin_out + nonlin_out
        return out
    
    #low-fidelity block alone
    def low_fidelity(self, ulf, y):
        return self.lf_subnet(ulf, y)
    
    # High-fidelity block, which includes both low-fidelity predictions and high-fidelity correlations (i.e., multifidelity)
    # See Equations (12)-(15) and Snippets 3-4 in the book chapter
    def high_fidelity(self, ulf, uhf, y):
        lf_out = jnp.array(self.lf_subnet(ulf, y)).reshape(-1,)
        lf_xhf = jax.vmap(self.lf_subnet, in_axes=(None,0))(ulf, self.xhigh).reshape(-1,)

        lin_out = self.lin_subnet(lf_out,y)
        nonlin_out = self.nonlin_subnet(jnp.hstack([uhf, lf_xhf]), y)
        out = lin_out + nonlin_out
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
            
#%% Standard mean squared-error loss functions for the low- and high-fidelity data    

# Low-fidelity loss: L_LF
@eqx.filter_jit  # compile this function to make it run fast.
def loss_low(model, data_low):
    u, y, v = data_low
    pred_v = jax.vmap(model.low_fidelity, in_axes=(0,0))(u,y)  # vectorise the model over a batch of data
    return optax.losses.l2_loss(pred_v, v).mean() #L2 loss

# High-fidelity loss: L_HF
@eqx.filter_jit  # compile this function to make it run fast.
def loss_high(model, data_high):
    ulf, uhf, y, v =  data_high
    pred_v = jax.vmap(model.high_fidelity, in_axes=(0,0,0))(ulf, uhf, y)  # vectorise the model over a batch of data
    return optax.losses.l2_loss(pred_v, v).mean() #L2 loss


# Total loss, includes low-fidelity, high-fidelity, and L2 regularization terms
# L_tot = lambda1*L_HF + lambda2*L_LF + lambda3*||W_nonlin||_2^2 + lambda4*||W_LF||_2^2
@eqx.filter_jit  # compile this function to make it run fast.
def loss(model, data_low, data_high):
    l_low = loss_low(model, data_low)
    l_high = loss_high(model, data_high)
    nonlin_weight = l2_reg(model.nonlin_subnet.branch)
    lf_weight = l2_reg(model.lf_subnet.branch)

    l_tot = l1*l_high + l2*l_low + l3*nonlin_weight + l4*lf_weight
    return l_tot

#total loss, without the regularization terms (for evaluation purposes)
@eqx.filter_jit  # compile this function to make it run fast.
def loss_eval(model, data_low, data_high):
    l_low = loss_low(model, data_low)
    l_high = loss_high(model, data_high)

    l_tot = l1*l_high + l2*l_low 
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
    train_data_low: list,  #low-fidelity training dataset
    train_data_high: list, #high-fidelity training dataset
    valid_data_low: list,  #low-fidelity validation dataset
    valid_data_high: list, #high-fidelity validation dataset
    batch_size_low: int,   #batch size for the low-fidelity dataset
    batch_size_high: int,  #batch size for the high-fidelity dataset
    key: Array,            #key for the random number generator for sampling batches
    steps: int,            #number of training steps
    print_every: int,      #frequency of priniting training stats
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
        data_high: list,
    ):

        # differentiate all floating-point arrays in `model`.
        grads = eqx.filter_grad(loss)(model, data_low, data_high)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state

    
    train_loss = evaluate(model, train_data_low, train_data_high)
    valid_loss = evaluate(model, valid_data_low, valid_data_high)
    print(
        f"\rtraining step={0}: train_loss={train_loss}, "
        f"validation_loss={valid_loss}"
    )
    
    train_loader_low_key, train_loader_high_key = jax.random.split(key, 2)
    train_loader_low = dataloader(train_data_low, batch_size_low, key=train_loader_low_key)
    train_loader_high = dataloader(train_data_high, batch_size_high, key=train_loader_high_key)
    for step in range(1, steps+1):

        #sample one batch from the low-fidelity training dataset
        data_low = next(train_loader_low)
        #sample one batch from the high-fidelity training dataset
        data_high = next(train_loader_high)

        #perfom a single optimization step
        model, opt_state = make_step(model, opt_state, data_low, data_high)
        
        #print training performance
        if (step % print_every) == 0 or (step == steps):
            train_loss = evaluate(model, train_data_low, train_data_high)
            valid_loss = evaluate(model, valid_data_low, valid_data_high)
            print(
                f"\rtraining step={step}: train_loss={train_loss}, "
                f"validation_loss={valid_loss}"
            )
            
    return model

def evaluate(model: DeepONet, data_low: list, data_high: list):
    """This function evaluates the model on the given dataset,
    computing the average loss and possibly any additional metrics.
    """
    # Note that all the JAX operations happen inside `loss`
    # 'loss' has JIT wrappers, so this is fast.
    avg_loss = loss_eval(model, data_low, data_high)
    return avg_loss
       
#%% Reference solutions to train and test the model

def low_fidelity_fun(a, y):
    u = a * y - 4
    if y <= 0.5:
        v = 0.5 * (6*y - 2)**2 * np.sin(u) + 10 * (y - 0.5) - 5
    else:
        v = 0.5 * (6*y - 2)**2 * np.sin(u) + 10 * (y - 0.5) - 2
    return v

def high_fidelity_fun(a, y):
    vlow = low_fidelity_fun(a, y)
    v = 2 * vlow - 20*y + 20
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

mlow = 40 #number of sampling points for input function
xlow = np.linspace(x0, x1, mlow) #sampling points for low-fidelity subnet
ylow = xlow #locations/coordinates of low-fidelity output evaluations

mhigh = 4 #number of sampling points for input function
xhigh = np.linspace(x0, x1, mhigh) #sampling points for high-fidelity (nonlinear) subnet
yhigh = xhigh #locations/coordinates of high-fidelity output evaluations

Nlow = 20   #number of input functions in the low-fidelity dataset
alow = np.linspace(a0, a1, Nlow)

Nhigh = 10  #number of input functions in the high-fidelity dataset
ahigh = np.linspace(a0, a1, Nhigh)

l1, l2 = 0.1, 1     #loss terms/weights for high- and low-fidelities
l3, l4 = 1e-1, 1e-3 #loss terms/weights for nonlinear and low-fideliy subnets (regularization)

#%% DeepONet architecture 

lf_branch_widths = [mlow, 40, 40, 40]  #for the low-fidelity networ, the branch takes the input function at mlow sample locations
lf_trunk_widths  = [1, 40, 40, 40]     #the trunk net takes a single coordinate (y)

lin_branch_widths = [1, 5] #for linear correlation subnet,  branch takes the low-fidelity prediction as input
lin_trunk_widths  = [1, 5] #the trunk net takes a single coordinate (y)

nonlin_branch_widths = [2*mhigh, 30, 30]  #for nonlinear correlation subnet,  branch takes the input function
                                          #at migh sample locations in addition to the low-fidelity predictions at these locations 
nonlin_trunk_widths  = [1, 30, 30]        #the trunk net takes a single coordinate (y)


#activation functions
lf_branch_activation = jax.nn.tanh
lf_trunk_activation = jax.nn.tanh
nonlin_branch_activation = jax.nn.tanh
nonlin_trunk_activation = jax.nn.tanh

#%% Generate datasets

#low-fidelity dataset
udata_low, ydata_low, vdata_low = [], [], [] #[u, y, v]
for a in alow:
    ulow = input_fun(a, xlow)  #sampling input functions at xlow locations for the low-fidelity dataset
    for y in ylow:
        vlow = low_fidelity_fun(a, y)
    
        udata_low.append(ulow)
        ydata_low.append(y)
        vdata_low.append(vlow)
        
lf_data = (jnp.array(udata_low), jnp.array(ydata_low).reshape([-1,1]), jnp.array(vdata_low))

    
#high-fidelity dataset
udata_high_l, udata_high_h, ydata_high, vdata_high = [], [], [], [] #[u, y, v]
for a in ahigh:
    ulow = input_fun(a, xlow)
    uhigh = input_fun(a, xhigh)
    for y in yhigh:
        vhigh = high_fidelity_fun(a, y)
    
        udata_high_l.append(ulow)
        udata_high_h.append(uhigh)
        ydata_high.append(y)
        vdata_high.append(vhigh)

hf_data = (jnp.array(udata_high_l), jnp.array(udata_high_h),
           jnp.array(ydata_high).reshape([-1,1]), jnp.array(vdata_high))

#%% Create training and validation datasets

dataset_size_low = len(udata_low)
train_data_size_low = int(dataset_size_low*0.8)

key, split_key = jax.random.split(key, 2)
indices = jax.random.permutation(split_key, dataset_size_low)
train_data_low = tuple(lf_data[i][indices[:train_data_size_low]] for i in range(3)) 
valid_data_low = tuple(lf_data[i][indices[train_data_size_low:]] for i in range(3)) 

dataset_size_high = len(udata_high_h)
train_data_size_high = int(dataset_size_high*0.8)

key, split_key = jax.random.split(key, 2)
indices = jax.random.permutation(split_key, dataset_size_high)
train_data_high = tuple(hf_data[i][indices[:train_data_size_high]] for i in range(4)) 
valid_data_high = tuple(hf_data[i][indices[train_data_size_high:]] for i in range(4)) 

#%% Build model

key, model_key = jax.random.split(key, 2)
model =  MFON(model_key,
              lf_branch_widths, lf_branch_activation, lf_trunk_widths, lf_trunk_activation,
              lin_branch_widths, lin_trunk_widths,
              nonlin_branch_widths, nonlin_branch_activation, nonlin_trunk_widths, nonlin_trunk_activation, xhigh)

#%% Model training

quick_mode = "--quick" in sys.argv
STEPS = 100 if quick_mode else 100_000 #number of training steps
LEARNING_RATE = 1e-3   #learning rate
batch_size_low = 32    #batch size for the low fidelity dataset
batch_size_high = 32   #batch size for the high fidelity dataset
PRINT_EVERY = 20 if quick_mode else 500 #frequency for printing training performance

if quick_mode:
    print(f"Running Example 3 in QUICK TEST mode (steps={STEPS}).")

## define optimizer
learning_rate = LEARNING_RATE
optim = optax.adamw(learning_rate) #Adam optimizer with weight decay regularization

key, loader_key = jax.random.split(key, 2)
model = train(model, optim,  
                       train_data_low, train_data_high,
                       valid_data_low, valid_data_high,
                       batch_size_low, batch_size_high,
                       loader_key, STEPS, PRINT_EVERY)

#%% Store model

outdir = './outputs/'
if not os.path.isdir(outdir):
    os.mkdir(outdir)

filename = outdir + 'Example3_model_mfon_mhigh_' + str(mhigh) + '_Nhigh_' + str(Nhigh) \
                  + '_mlow_' + str(mlow) + '_Nlow_' + str(Nlow) + '.eqx'
eqx.tree_serialise_leaves(filename, model)

#%% Test the model

atest = 12.0 #value of (a) for testing
ntest = 100  #number of evaluation points to plot the DeepONet prediction vs true solution
             #this shows that once trained, an operator network can be queried at any locations
ytest = np.linspace(x0, x1, ntest)


#the input function for the test case
#note that sampling *locations* should be the same as in training
utest_low = input_fun(atest, xlow)
utest_high = input_fun(atest, xhigh)

#true solution
vtest = np.zeros(ntest)
for i, y in enumerate(ytest):
    vtest[i] = high_fidelity_fun(atest, y)

#MFON predictions
vpred = jax.vmap(model, in_axes=(None, None, 0))(utest_low, utest_high, ytest.reshape(-1,1))

#relative l2 error
err = jnp.linalg.norm(vtest-vpred)/jnp.linalg.norm(vtest)
print(f"Relative l2 error is {err}")

#Low-fidelity subnet output
vpred_low = jax.vmap(model.low_fidelity, in_axes=(None, 0))(utest_low, ytest.reshape(-1,1))

#high fidelity test data points (at the same locations in training, included for reference)
vtesthigh = np.zeros_like(yhigh)
for i, y in enumerate(yhigh):
    vtesthigh[i] = high_fidelity_fun(atest, y)
    
#low-fidelity test data points
vtestlow = np.zeros_like(ylow)
for i, y in enumerate(ylow):
    vtestlow[i] = low_fidelity_fun(atest, y)


#%% Plotting results
fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(8,4))

ax.plot(ytest, vtest, label=r'\bf{Exact}', color = 'C0', linewidth=2)
ax.plot(ytest, vpred, label=r'\bf{MFON Prediction}', linestyle='-', color = 'C3', linewidth=2)#, zorder=2)
ax.plot(ytest, vpred_low, label=r'\bf{Low-Fidelity Subnet}', linestyle='--', color = 'C1', linewidth=2)

ax.plot(yhigh, vtesthigh, label=r'\bf{Reference High-Fidelity Data Points}', linestyle='None', marker='o', markersize=8, color = 'C2')
ax.plot(ylow, vtestlow, label=r'\bf{Reference Low-Fidelity Data Points}', linestyle='None', marker='o', markersize=8, color = 'C7')

ax.set_xlabel(r'$y$', fontsize=18)
ax.set_ylabel(r'$v$', fontsize=18)

ax.legend(ncol=2, fontsize=10, handletextpad=0.5, columnspacing=1.5, handlelength=3)
fig.subplots_adjust(hspace=0.9, wspace=0.35)

plotdir = './plots/'
if not os.path.isdir(plotdir):
    os.mkdir(plotdir)
        
plt.savefig(plotdir+'Example3_MFON_jump_function.png', dpi = 500, bbox_inches = 'tight')
plt.savefig(plotdir+'Example3_MFON_jump_function.pdf', dpi = 500, bbox_inches = 'tight')