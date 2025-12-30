# Bayesian Estimation of Causal Effects Using Proxies of a Latent Interference Network

**Authors:** Bar Weinstein, Daniel Nevo  
**Department of Statistics and Operations Research, Tel Aviv University**

[![arXiv](https://img.shields.io/badge/arXiv-2505.08395-b31b1b.svg)](https://arxiv.org/abs/2505.08395)

---

## Overview

This repository contains the implementation of the framework proposed in the manuscript.

Network interference occurs when the treatment of one unit affects the outcome of others. In many practical settings, the true interference network is unobserved (latent), and researchers only have access to **proxy networks** (noisy measurements, multiple data sources, or multilayer networks). 

This codebase implements a **Bayesian inference framework** to estimate causal effects in this setting. It employs a Structural Causal Model (SCM) that accommodates diverse proxy types and utilizes a **Block Gibbs sampler** to jointly recover the latent network structure and estimate causal parameters.
The latent network is updated using **Locally Informed Proposals (LIP)**.

This repository includes scripts for both **fully synthetic experiments** and **semi-synthetic experiments**.
It also contains all necessary components to reproduce the results presented in the manuscript.


## Repository Structure

The code is organized as follows:

```bash

├── Data/
│   └── cs_aarhus/           # Scripts for semi-synthetic experiments (Aarhus data)
│       └── combined_analysis.py # Run the full analysis 
│       └── data_mcmc.py # MCMC samplers used
│       └── data_models.py # Probabilistic models used
│       └── util_data.py # Utility functions 
│       └── results_analysis.ipynb # summary and plots for the results
│       └── cs_analysis_results.csv # Results data file
│       └── figs/ # Figures of the results
├── Simulations/             # Scripts for fully synthetic experiments
│   ├── mwg_simulations.py   # Main script that runs the simulations
│   ├── data_gen.py          # Data generation functions
│   └── simulation_aux.py    # Helper loops for simulation iterations
│   └── plot_results.ipynb   # Plotting and summarizing simulation results
│   └── additional_figure.ipynb   # Additional figure for the supplement
│   └── combined_sim_results.csv   # Combined simulation results data file
│   └── mwg_scaling_results.csv   # scaling analysis results data file
│   └── figs/ # Figures of the results
├── src/                     # Core methodology implementation
│   ├── GWG.py               # Locally Informed Proposals (GWG) implementation
│   ├── MWG_sampler.py       # The Block Gibbs Sampler (init and main sampler)
│   ├── MCMC_conti_relax.py  # MCMC with continuous relaxation of A*
│   ├── MCMC_fixed_net.py    # MCMC with fixed network 
│   ├── Models.py            # probabilistic models
│   └── utils.py             # Utility functions
├── requirements.txt         # Python dependencies
└── README.md

```

## Installation & Requirements

The codebase is written in **Python 3.13**. 

To install the required dependencies, run:

```bash 
pip install -r requirements.txt
```

## Usage
*Note:* The experiments ran on a powercluster with multiple cores. Adjust the number of iterations and cores as needed.
### Fully Synthetic Experiments
To reproduce the fully synthetic experiments described in the paper:

1.  Navigate to the project root.
2.  Adjust configuration variables (e.g., `N_CORES`) in `Simulations/mwg_simulations.py` if necessary.
3.  Run the module:

```bash
python -m Simulations.mwg_simulations
```


### Semi-Synthetic Experiments
To reproduce the semi-synthetic experiments using the Aarhus dataset:
1.  Navigate to the project root.
2.  Run the combined analysis script:

```bash
python Data/cs_aarhus/combined_analysis.py
```
The data used in the semi-synthetic experiments is available at: https://manliodedomenico.com/data.php


## Citation
If you use this code for your research, please cite the following preprint:

```
@article{weinstein2025bayesian,
  title={Bayesian Estimation of Causal Effects Using Proxies of a Latent Interference Network},
  author={Weinstein, Bar and Nevo, Daniel},
  journal={arXiv preprint arXiv:2505.08395},
  year={2025}
}
```