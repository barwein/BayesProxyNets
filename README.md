# Estimating Causal Effects Using Proxies of the Interference Network

See preprint at (arXiv link here).

## Code Structure

```mermaid
graph TD
    A[Project Root]
    A --> B[src/]
    A --> C[Simulations/]
    A --> D[Data/]
    B --> B1[GWG.py]
    B --> B2[MWG_sampler.py]
    B --> B3[Models.py]
    B --> B4[MCMC_fixed_net.py]
    B --> B5[utils.py]
    C --> C1[mwg_simulatons.py]
    C --> C2[data_gen.py]
    C --> C3[simulation_aux.py]
    C --> C4[Networks_graphic.ipynb]
    C --> C5[plot_results.ipynb]
    C --> C6[Sim_results_analysis.ipynb]
    C --> C7[additional_figure.ipynb]
    D --> D1[cs_aarhus/]
    D1 --> D1a[combined_analysis.py]
    D1 --> D1b[cs_a_analysis.ipynb]
    D1 --> D1c[data_mcmc.py]
    D1 --> D1d[data_models.py]
    D1 --> D1e[util_data.py]
    D1 --> D1f[results_analysis.ipynb]
```

This diagram provides a high-level overview of the main code and data structure. Notebooks are marked with `.ipynb` extensions.

The code is organized as follows:
- `src`: contains auxiliary scripts used in the numerical illustrations.
- `Simulations/`: contains the code to run the fully-synthetic experiments.
- `Data/cs_aarhus`: contains the code to run the semi-synthetic experiments.

## Software Requirements
The code is written in Python 3.11. Run 'requirements.txt' file to install all required packages. 

## Running the Experiments
- The script `Simulations/mwg_simulations.py` contains the code to run the fully-synthetic experiments.
- The script `Data/cs_aarhus/combined_analysis.py` contains the code to run the semi-synthetic experiments. 

Both scripts were executed in a power-cluster with multiple cores. Running them in a PC with CPU might take a while. Adjust parameters accordingly!

## Additional Files

- The file `src/GWG.py` contain the implementation of the Locally Informed Proposals with gradient approximations.
- The file `src/MWG_sampler.py` contain the implementation of the Block Gibbs algorithm.

## Data Availability
The data used in the semi-synthetic experiments is available at: https://manliodedomenico.com/data.php
