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

The code is organized to separate core logic, probabilistic models, and simulation execution:

```bash
.
├── Data/
│   └── cs_aarhus/           # Scripts for semi-synthetic experiments (Aarhus data)
│       └── combined_analysis.py
├── Simulations/             # Scripts for fully synthetic experiments
│   ├── mwg_simulations.py   # Main entry point for synthetic simulations
│   ├── data_gen.py          # Synthetic data generation (DGP)
│   └── simulation_aux.py    # Helper loops for simulation iterations
├── src/                     # Core methodology implementation
│   ├── GWG.py               # Locally Informed Proposals (GWG) implementation
│   ├── MWG_sampler.py       # The Block Gibbs Sampler logic
│   ├── Models.py            # NumPyro probabilistic models (Outcome, Proxy, Priors)
│   └── utils.py             # Utility functions (exposures, error metrics)
├── main.pdf                 # The manuscript
├── requirements.txt         # Python dependencies
└── README.md

```