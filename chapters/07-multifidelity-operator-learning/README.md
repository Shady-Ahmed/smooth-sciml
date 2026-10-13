# Chapter 7: Multifidelity Operator Learning

**Authors:** Amanda A. Howard, Panos Stinis, and Shady E. Ahmed  
*Pacific Northwest National Laboratory (PNNL), Richland, WA 99354*  
**Contact:** `amanda.howard@pnnl.gov`, `panagiotis.stinis@pnnl.gov`, `shady.ahmed@pnnl.gov`

This directory provides the companion code for **Chapter 7: Multifidelity Operator Learning** of the edited volume *A Smooth Introduction to Scientific Machine Learning: Mathematical Foundations and Numerical Implementation*, edited by Victorita Dolean, Alexander Heinlein, and Olga Mula.

---

## Overview

Operator learning aims to learn mappings between infinite-dimensional function spaces rather than finite-dimensional vectors. This chapter presents deep operator networks (DeepONets) and multifidelity operator networks (MFON), combining datasets and physics models across disparate fidelity levels:

- **Example 1**: Data-driven DeepONet for a 1D discontinuous jump function.
- **Example 2**: Physics-informed DeepONet (PI-DeepONet) for the 1D viscous Burgers' equation without high-fidelity solution data.
- **Example 3**: Data-driven multifidelity operator network (MFON) combining abundant low-fidelity data with scarce high-fidelity data for the jump function.
- **Example 4**: Multifidelity physics-informed DeepONet combining low-fidelity simulation data with high-fidelity physical governing laws (viscous Burgers' equation).
- **Example 5**: Multifidelity closure modeling for Galerkin proper orthogonal decomposition (POD) reduced-order models (GROM) at high Reynolds numbers ($\text{Re} = 15{,}000$).

---

## Software & Hardware Requirements

- **Python Version**: Tested on Python **3.10** and **3.11** (recommended: Python 3.11).
- **Hardware**: All examples can be executed on standard modern consumer hardware (laptop/workstation CPU, 8 GB+ RAM).
  - GPU acceleration (NVIDIA CUDA or Apple Metal via JAX) is supported and will speed up training, but is strictly optional.
- **Dependencies**: `jax`, `equinox`, `optax`, `jaxtyping`, `numpy`, `scipy`, `matplotlib`, `tqdm`. (Notice: No external heavy frameworks like PyTorch are required; all data batching and network operations use native JAX).

---

## Environment Setup

### Option 1: Conda Environment (Recommended)

To create a clean environment using the provided `environment.yml`:

```bash
# Navigate to the chapter folder
cd chapters/07-multifidelity-operator-learning

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate sciml-ch07
```

### Option 2: Pip in a Virtual Environment

Alternatively, using Python's built-in `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Examples

Each example folder is self-contained. Below are the commands to execute each example, along with approximate execution times on a standard multi-core laptop CPU.

> **Tip for Quick Verification**: All training scripts accept a `--quick` command-line flag (e.g. `python Example1-DON-data-only-jump-function.py --quick`) which trains for 100 iterations. This verifies that JIT compilation, network forward/backward passes, and plot saving run end-to-end in just a few seconds.

---

### Example 1: Data-Driven DeepONet (Jump Function)

Trains a standard DeepONet on pairs of input function samples and spatial query coordinates for a 1D discontinuous jump function.

```bash
cd Example1-DON-Data-only
python Example1-DON-data-only-jump-function.py
```
- **Approximate runtime**: ~1.5 minutes (100,000 steps on CPU) | < 5 seconds with `--quick`.
- **Output**: Trained model in `outputs/` and comparison plot in `plots/Example1_DON_jump_function.png`.

---

### Example 2: Physics-Informed DeepONet (1D Burgers)

First generates the high-fidelity reference trajectories using a spectral Fourier solver, and then trains a PI-DeepONet enforcing initial condition, periodic boundary condition, and viscous Burgers PDE residual.

```bash
cd ../Example2-DON-Physics-only

# Step 1: Generate reference solutions (RK45 spectral solver)
python Example2-Burgers-reference-solutions.py

# Step 2: Train or evaluate PI-DeepONet
python Example2-DON-physics-only-Burgers.py --train
```
- **Approximate runtime**: 
  - Reference generation: ~15 seconds.
  - Training (100,000 steps): ~8 minutes on CPU | ~10 seconds with `--quick`.
- **Output**: Reference datasets in `burgers-data/`, model in `outputs/`, and solution field in `plots/`.

---

### Example 3: Multifidelity DeepONet (Data-Driven Jump Function)

Demonstrates how a composite MFON (low-fidelity subnet + linear correlation subnet + nonlinear correlation subnet) leverages coarse low-fidelity data ($m=40, N=20$) to dramatically boost performance when high-fidelity data is severely scarce ($m=4, N=10$).

```bash
cd ../Example3-MFON-Data-only
python Example3-MFON-data-only-jump-function.py
```
- **Approximate runtime**: ~2 minutes on CPU | < 5 seconds with `--quick`.
- **Output**: Model in `outputs/` and multi-fidelity comparison in `plots/Example3_MFON_jump_function.png`.

---

### Example 4: Multifidelity Physics-Data DeepONet (1D Burgers)

Combines coarse numerical simulation data (low-fidelity) with continuous governing PDE residuals (high-fidelity physics) to accurately predict shock formation without expensive high-fidelity solution data.

```bash
cd ../Example4-MFON-Physics-Data

# Step 1: Generate high- and low-fidelity reference solutions
python Example4-Burgers-reference-solutions-high-and-low.py

# Step 2: Train multifidelity physics-informed network
python Example4-MFON-physics-low-data-Burgers.py --train
```
- **Approximate runtime**:
  - Reference generation: ~20 seconds.
  - Training: ~5 minutes on CPU | ~10 seconds with `--quick`.
- **Output**: Reference data in `burgers-data/`, model in `outputs/`, and comparison plots in `plots/`.

---

### Example 5: Multifidelity ROM Closure at High Reynolds Numbers

Addresses the closure modeling problem in nonlinear Galerkin reduced-order models (GROM) for Burgers' equation across a range of Reynolds numbers ($\text{Re} \in [2500, 10000]$) and tests extrapolation at $\text{Re} = 15000$.

```bash
cd ../Example5-MFON-Closure-GROM

# Step 1: Full-order model (FOM) simulation data generation
python Example5_1-burgers-fom-data-generation.py

# Step 2: Proper Orthogonal Decomposition (POD) and standard Galerkin ROM
python Example5_2-burgers-grom-low-fidelity.py

# Step 3: Multifidelity DeepONet in-the-loop closure training and testing
python Example5_3-burgers-grom-mutli-fidelity.py
```
- **Approximate runtime**:
  - FOM generation (Step 1): ~2 minutes on CPU.
  - POD & ROM (Step 2): ~20 seconds.
  - DeepONet Closure (Step 3): ~6 minutes on CPU | ~15 seconds with `--quick`.
- **Output**: FOM snapshots and precomputed Galerkin tensors in `data/`, plots in `plots/`.

---

## Quick Verification Mode & Workspace Cleaning

### Quick-Test Flag (`--quick`)

All example scripts accept an optional `--quick` flag to enable rapid end-to-end sanity testing:
- **Examples 1–4 and 5_3**: Reduces the number of optimization steps from 10,000–100,000 to **100 steps**, allowing the full training loop, loss evaluation, JIT compilation, and plot generation to complete in a few seconds.
- **Example 5_1**: In `--quick` mode, generates data for a reduced set of 2 training Reynolds numbers ($\text{Re} \in [5000]$) and the test case ($\text{Re} = 15000$), cutting FOM generation time in half while preserving numerical accuracy and array shapes.
- **Example 5_2 & 5_3**: Automatically adapt to the available training Reynolds numbers generated by 5_1 (2 in quick mode, 4 in full mode).
- **Model Checkpoints**: Checkpoints produced in `--quick` mode are saved with their step count in the filename (e.g. `_epochs100.eqx`), ensuring full-fidelity production models are never overwritten.

### Workspace Reset (`clean.py`)

To remove all generated simulation datasets, checkpoints, models, and plot files before a fresh run or git commit:

```bash
# From the chapter root directory
python clean.py
```

---

## Citation

If you use this companion code or find the methodology helpful in your research, please cite:

```bibtex
@incollection{howard2026multifidelity,
  title     = {Multifidelity Operator Learning},
  author    = {Howard, Amanda A. and Stinis, Panos and Ahmed, Shady E.},
  booktitle = {A Smooth Introduction to Scientific Machine Learning: Mathematical Foundations and Numerical Implementation},
  editor    = {Dolean, Victorita and Heinlein, Alexander and Mula, Olga},
  publisher = {Springer},
  year      = {2026}
}
```

Metadata is also provided in machine-readable CFF format in `CITATION.cff`.

---

## License & Disclaimer

The companion code for this chapter is provided by Pacific Northwest National Laboratory (operated by Battelle for the U.S. Department of Energy under Contract DE-AC05-76RL01830) under the **BSD 2-Clause License** (Copyright Battelle Memorial Institute 2025) and is incorporated into this companion repository under the terms of the **GNU General Public License v3.0** (GPL-3.0-only).

See the [LICENSE](LICENSE) file for the full text of the Battelle license and U.S. Government disclaimer.
