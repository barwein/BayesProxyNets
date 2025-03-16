# Estimating Causal Effects Using Proxy Interference Networks

Replication files for ``Estimating Causal Effects Using Proxies of the Interference Network``.
See preprint at (arXiv link here).

## Code Structure
The code is organized as follows:
- `src`: contains auxiliary scripts used in the numerical illustrations.
    -- test
- `Simulations/`: contains the code to run the simulations.
- `hsgp/`: contains the code to run the HSGP algorithm based on [NumPyro implementation](https://github.com/pyro-ppl/numpyro/tree/master/numpyro/contrib/hsgp).
- `Data_analysis/`: contains the code to run the data analysis on the [Paluck et el. (2016) study](https://www.pnas.org/doi/abs/10.1073/pnas.1514483113#data-availability).

## Software Requirements
The code is written in Python 3.11. The following packages are required to run the code:
- numpy, jax, torch, numpyro, pyro, scipy, pandas, matplotlib, seaborn, networkx, tqdm, os, itertools, time, pyreadr

## Running the Code
- The script `Simulations/linear_dgp_simulations.py` contains the code to run the simulations.

- The script `Data/Palluck_et_al/run_analysis` contains the code to run the data analysis. 


Both scripts were executed in a power-cluster with multiple cores. Running them in a PC might take a while. Adjust parameters accordingly!

## Data Availability
The data used in the data analysis is available at the following link: https://www.icpsr.umich.edu/web/ICPSR/studies/37070
