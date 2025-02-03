import jax.numpy as jnp
from jax import random
from jax.scipy.special import logit
import numpy as np

from Simulations.simulation_aux import one_simulation_iter
import Simulations.data_gen as dg
import src.utils as utils

import os


# --- Set cores and seed ---
N_CORES = 4
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_CORES}"

# --- Global variables ---

N = 500
TRIU_DIM = N * (N - 1) // 2

THETA = jnp.array([-2.5, 1])
GAMMA = jnp.array([logit(0.85), logit(0.1), 1])

print("GAMMA", GAMMA)   

ETA = jnp.array([-1, 3, -0.25, 2])
SIG_INV = 2 / 3
RHO = 0.5
PZ = 0.5

param = utils.ParamTuple(theta=THETA, gamma=GAMMA, eta=ETA, rho=RHO, sig_inv=SIG_INV)

# test one iter

FILEPATH = "Simulations/results"

rng_key = random.PRNGKey(0)
rng = np.random.default_rng(15)

# gen data

fixed_data = dg.generate_fixed_data(rng, N, param, PZ)
proxy_nets = dg.generate_proxy_networks(
    rng, TRIU_DIM, fixed_data["triu_star"], GAMMA, fixed_data["x_diff"], fixed_data["Z"]
)

data_sim = dg.data_for_sim(fixed_data, proxy_nets)

print("true exposure mean", data_sim.true_exposures.mean())

new_interventions = dg.new_interventions_estimands(
    rng, N, fixed_data["x"], fixed_data["triu_star"], ETA
)

# run one simulation

one_simulation_iter(
    idx=1,
    rng_key=random.split(rng_key)[0],
    data=data_sim,
    new_interventions=new_interventions,
    cur_gamma_noise=GAMMA[2],
    file_path=FILEPATH,
)
