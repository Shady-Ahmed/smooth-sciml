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

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#%% Extra platform settings

# Global flag to set a specific platform, must be used at startup.
# more details: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
jax.config.update('jax_platform_name', 'cpu') #change to 'gpu', if available
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

#%%

# Burgers right-hand side du/dt = -u du/dx + \nu d^2u/dx^2
def burgers_rhs_spectral(t, u, dx):
    nx = u.shape[0]
    # Spectral wavenumbers
    k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ik = 1j * k
    k2 = k ** 2
    
    # Fourier transform
    u_hat = np.fft.fft(u)
    
    # Spectral derivatives
    dudx = np.fft.ifft(ik * u_hat).real 
    d2udx2 = np.fft.ifft(-k2 * u_hat).real 
    return -u * dudx + nu * d2udx2

# initial condition functions, see Eq. 10 in the book chapter
def smooth_perturbed_sine(x, key, modes=3):
    base = np.sin(2 * jnp.pi * x)
    
    key = jax.random.PRNGKey(0) if key is None else key
    a_key, b_key = jax.random.split(key)
    a = np.asarray(jax.random.normal(a_key, shape=(modes,)))
    perturb = sum(a[k] * jnp.sin((k+1) * 2 * jnp.pi * x)  for k in range(modes)) * 0.1 #0.2
    
    return base + perturb

# Generate a batch of initial conditions to prepare the dataset
def generate_batch_initial_conditions(x, key, dataset_size, modes=3):
    keys = jax.random.split(key, dataset_size)
    return np.stack([smooth_perturbed_sine(x, key, modes) for key in keys])

# Solve Burgers problem for a given IC
def solve_single_ic(u0, rhs):
    u0 = np.asarray(u0[:-1]) #remove last point for periodic BC treatment
    sol = solve_ivp(rhs, t_span, u0, t_eval=t_eval, method='RK45')
    usol = sol.y.T
    #recover periodicity
    usol = np.hstack([usol,usol[:,0].reshape(-1,1)])
    return usol

# Solve batch serially: call solve_single_ic for different initial conditions
def solve_batch(batch_u0, rhs):
    results = [solve_single_ic(u0, rhs) for u0 in batch_u0]
    return results

#%% Problem Setup 

nu = 1e-2 #kinematic viscosity

lx = 1.0 #domain size
nx_hf = 256  #number of grid points in x direction for high-fidelity is nx_hf+1
nx_lf = 8    #number of grid points in x direction for low-fidelity is nx_lf+1

x_hf = jnp.linspace(0, lx, nx_hf+1, endpoint=True)
x_lf = jnp.linspace(0, lx, nx_lf+1, endpoint=True)

#spatial grid spacing for low- and high-fidelity solutins
dx_lf = lx / nx_lf
dx_hf = lx / nx_hf

t_span = (0.0, 1.0)
nt = 101 #number of time instants at which solution is stored
t_eval = np.linspace(*t_span, nt)

key = jax.random.PRNGKey(42)

quick_mode = "--quick" in sys.argv
if quick_mode:
    print("Running Example 4 reference solutions generation in QUICK mode.")

dataset_size_hf = 10 #number of initial conditions to build reference high-fidelity dataset
u0_batch_hf = generate_batch_initial_conditions(x_hf, key, dataset_size=dataset_size_hf, modes=3)
rhs_hf = lambda t,u: burgers_rhs_spectral(t, u, dx_hf)
solutions_hf = solve_batch(u0_batch_hf, rhs_hf)


dataset_size_lf = 10 #number of initial conditions to build low-fidelity dataset
u0_batch_lf = generate_batch_initial_conditions(x_lf, key, dataset_size=dataset_size_lf, modes=3)
rhs_lf = lambda t,u: burgers_rhs_spectral(t, u, dx_lf)
solutions_lf = solve_batch(u0_batch_lf, rhs_lf)

#%% Save data

outdir = './burgers-data/'
if not os.path.isdir(outdir):
    os.mkdir(outdir)

filename = outdir + 'Example4_burgers_periodic_' + str(dataset_size_hf) + '_ICS_high.npy'
np.save(filename, np.stack(solutions_hf))

filename = outdir + 'Example4_burgers_periodic_' + str(dataset_size_lf) + '_ICS_low.npy'
np.save(filename, np.stack(solutions_lf))

#%% Plot example solution from batch
u_vals = solutions_hf[0]
X, T = np.meshgrid(x_hf, t_eval)

plt.figure(figsize=(8, 4))
plt.contourf(X, T, u_vals, levels=50, cmap='turbo')
plt.colorbar(label='u(x,t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Burgers Equation with Smooth Perturbed Sine Initial Condition')
plt.tight_layout()
plt.savefig(outdir + 'Example4_reference_solution.png', dpi=200, bbox_inches='tight')
if not quick_mode:
    plt.show()
plt.close()
