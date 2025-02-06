###
# This script runs the simulations for the linear model with CAR cov
###

import jax.numpy as jnp
from jax import random
from jax.scipy.special import logit
import numpy as np

from Simulations.simulation_aux import one_simulation_iter
import Simulations.data_gen as dg

import os


# --- Set cores and seed ---
N_CORES = 4
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_CORES}"

# --- Global variables ---

N = 500
# N = 1000
TRIU_DIM = N * (N - 1) // 2

THETA = jnp.array([-2, 1])
# GAMMA_FIX = jnp.array([logit(0.9), logit(0.1), 1])
GAMMA_FIX = jnp.array([logit(0.9), logit(0.1)])
# GAMMA_X_NOISES = jnp.arange(0.2, 2.2, 0.2)
GAMMA_X_NOISES = jnp.arange(0.2, 1.1, 0.1)

ETA = jnp.array([-1, 3, -0.25, 2])
SIG_INV = 2 / 3
RHO = 0.5
PZ = 0.5

# param = utils.ParamTuple(theta=THETA, gamma=GAMMA, eta=ETA, rho=RHO, sig_inv=SIG_INV)
PARAM = {
    "theta": THETA,
    "eta": ETA,
    "rho": RHO,
    "sig_inv": SIG_INV,
}
FILEPATH = "Simulations/results"

N_ITER = 1
N_GAMMAS = GAMMA_X_NOISES.shape[0]

# TODO: check why MWG_init with svi in network module is not stable and not working that well!

for i in range(N_ITER):
    # Set keys
    rng_key = random.PRNGKey(i)
    rng = np.random.default_rng(i)

    # gen data (not depedent on gamma)
    fixed_data = dg.generate_fixed_data(rng, N, PARAM, PZ)

    print(f"mean true exposures: {jnp.mean(fixed_data['true_exposures'])}")

    # gen new interventions
    new_interventions = dg.new_interventions_estimands(
        rng, N, fixed_data["x"], fixed_data["triu_star"], ETA
    )

    for j in range(N_GAMMAS):
        print(f"### iteration {i}, gamma noise {j} ###")

        # update gamma
        # param["gamma"] = jnp.append(GAMMA_FIX, GAMMA_X_NOISES[i])
        cur_gamma = jnp.append(GAMMA_FIX, GAMMA_X_NOISES[j])
        print(f"cur_gamma: {cur_gamma}")

        # sample proxy networks with current gamma
        proxy_nets = dg.generate_proxy_networks(
            rng,
            TRIU_DIM,
            fixed_data["triu_star"],
            cur_gamma,
            fixed_data["x_diff"],
            fixed_data["Z"],
        )

        data_sim = dg.data_for_sim(fixed_data, proxy_nets)

        # run one iteration
        rng_key = random.split(rng_key)[0]
        idx = i * N_ITER + j * N_GAMMAS

        one_simulation_iter(
            idx=idx,
            rng_key=rng_key,
            data=data_sim,
            new_interventions=new_interventions,
            cur_gamma_noise=GAMMA_X_NOISES[j],
            file_path=FILEPATH,
        )
