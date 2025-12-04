###
# This script runs the simulations for the linear model with CAR cov
###

# --- Import libraries ---

import jax.numpy as jnp
from jax import random
from jax.scipy.special import logit

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Simulations.simulation_aux import one_simulation_iter
import Simulations.data_gen as dg

# --- Set cores and seed ---

N_CORES = 4  # update accordingly, can use GPU as well
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_CORES}"

# --- Global variables ---

N = 500
TRIU_DIM = N * (N - 1) // 2

THETA = jnp.array([-2.5, 1])
GAMMA_BASELINE = jnp.array([logit(0.95), logit(0.05)])

GAMMA_REP = jnp.array([logit(0.8), 1.5, logit(0.2), 1.5])
GAMMA_X_NOISES = jnp.arange(2, 4 + 1e-6, 0.5)

GAMMA_B_NOISE_0 = GAMMA_BASELINE[0] - GAMMA_X_NOISES / 2
GAMMA_B_NOISE_1 = GAMMA_BASELINE[1] + GAMMA_X_NOISES / 2

# ETA = jnp.array([-1, 3, -0.5, 2])
# ETA = jnp.array([-1, 3, -0.5, 0.25])
ETA = jnp.array([-1, 3, -0.5, 1])
SIG_INV = 1.0
RHO = 0.5
PZ = 0.5

PARAM = {
    "theta": THETA,
    "eta": ETA,
    "rho": RHO,
    "sig_inv": SIG_INV,
}
FILEPATH = "Simulations/results"

#
N_ITER = 1
# N_ITER = 300
N_GAMMAS = GAMMA_X_NOISES.shape[0]


if __name__ == "__main__":
    for i in range(N_ITER):
        # Set keys
        rng_key = random.PRNGKey(i * 55 + 1)

        # generate data (not depedent on gamma)
        rng_key, _ = random.split(rng_key)
        fixed_data = dg.generate_fixed_data(rng_key, N, PARAM, PZ)

        # true_vals for wasserstein distance
        true_vals = {
            "eta": ETA,
            "rho": jnp.array([RHO]),
            "sig_inv": jnp.array([SIG_INV]),
            "triu_star": fixed_data["triu_star"],
        }

        print(f"mean true exposures: {jnp.mean(fixed_data['true_exposures'])}")

        # generate new interventions
        rng_key, _ = random.split(rng_key)
        new_interventions = dg.new_interventions_estimands(
            rng_key, N, fixed_data["x"], fixed_data["triu_star"], ETA
        )

        for j in range(N_GAMMAS):
            print(f"### iteration {i}, gamma noise {j} ###")

            # update gamma
            cur_gamma = jnp.concatenate(
                [
                    jnp.array([GAMMA_B_NOISE_0[j]]),
                    jnp.array([GAMMA_B_NOISE_1[j]]),
                    jnp.array([GAMMA_X_NOISES[j]]),
                    GAMMA_REP,
                ]
            )

            print("cur gamma: ", cur_gamma)

            rng_key = random.split(rng_key)[0]
            # sample proxy networks with current gamma
            proxy_nets = dg.generate_proxy_networks(
                # rng,
                rng_key,
                TRIU_DIM,
                fixed_data["triu_star"],
                cur_gamma,
                fixed_data["x_diff"],
                fixed_data["Z"],
            )

            data_sim = dg.data_for_sim(fixed_data, proxy_nets)

            # run one iteration
            rng_key = random.split(rng_key)[0]
            # a bit different in final results as it account for cluster job_id
            idx = str(i) + "_" + str(j)

            one_simulation_iter(
                iter=i,
                idx=idx,
                rng_key=rng_key,
                data=data_sim,
                new_interventions=new_interventions,
                cur_gamma_noise=GAMMA_X_NOISES[j],
                true_vals=true_vals,
                file_path=FILEPATH,
            )
