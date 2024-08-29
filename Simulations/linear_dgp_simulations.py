import pandas as pd
import multiprocessing
import jax
import jax.numpy as jnp
from jax import random, vmap
import time
import numpy as np
import src.Aux_functions as aux
from Simulations import vectorized_simulations, results_to_pd_df, one_simuation_iter
import argparse

# parameters guides:
# theta: p(A* | X, theta)
# gamma: p(A | A*, X, gamma)
# eta, sig_y: p(Y | Z, X, A*, eta, sig_y)
# alpha: pi_alpha(z) ---> stochastic intervention

# --- Define global variables ---
RANDOM_SEED = 62625235
rng = np.random.default_rng(RANDOM_SEED)

# THETA = [-2, -0.5]
# THETA = jnp.array([-2, -0.5, 2])
# THETA = jnp.array([-2, -.5, 2])
THETA = [-2, 1.5]
# GAMMA = jnp.array([0.05, 0.3])
# GAMMA = jnp.array([0.75, -2.5, -1])
# GAMMA = jnp.array([0.75, -1])
GAMMA = [1.1, 0.2,  -1]
# GAMMA = [2.5, -1.5]
# GAMMA = [1.5, -1.5]
# ETA = jnp.array([-1, 3, -0.25, 2.5])
ETA = jnp.array([-1, 3, -0.25, 1, 2.5])
# ETA = jnp.array([-1, 3, -0.25, 0, 2.5])
# ETA = jnp.array([-1, 3, 0, 0, 2.5])
SIG_Y = 1
PZ = 0.5
LIN_Y = True
ALPHAS = (0.7,0.3)
N = 500
# N = 300
# N_SIM = 500
N_SIM = 1
# N_REP = 1000
N_REP = 100

if __name__ == "__main__":
    print("### Starting simulation ###")
    print("N_CORES: ", multiprocessing.cpu_count())
    print("N jax cpu devices: ", jax.local_device_count())

    # parser = argparse.ArgumentParser(description="Run simulation with a specific job ID")
    # parser.add_argument('job_id', type=int, help='Job ID to run the simulation')
    # args = parser.parse_args()
    # if args.job_id is None:
    #     raise ValueError("Error: No job id is given.")

    start = time.time()
    idx_range = jnp.arange(N_SIM)
    # (idx, theta, gamma, eta, sig_y, pz, n_rep, lin_y, alpha, w_path, dgp)
    # Run simulations
    # for i in range(N_SIM):
    #     sim_results = one_simuation_iter(i, THETA, GAMMA, ETA, SIG_Y, PZ, N_REP, LIN_Y, ALPHA)
    #     df_results = results_to_pd_df(sim_results, i)
    #     w_path = "results"
    #     dgp = "linear_dgp"
    #     with_header = i == 0
    #     df_results.to_csv(w_path + "/" + dgp + "_" + args.job_id + ".csv",
    #                       mode='a', index=False, header=with_header)
        # df_results.to_csv(w_path + "/" + dgp + ".csv", index=False)


    # sim_results = vectorized_simulations(idx_range, THETA, GAMMA,
    #                                      ETA, SIG_Y, PZ,
    #                                      N_REP, LIN_Y, ALPHAS)
    sim_results = one_simuation_iter(35, THETA, GAMMA,
                                         ETA, SIG_Y, PZ,
                                         N_REP, LIN_Y, ALPHAS)
    df_results = results_to_pd_df(sim_results, N_SIM)
    print(df_results)
    # combined_df = pd.concat(sim_results, ignore_index=True)
    w_path = "results"
    dgp = "linear_dgp_test"
    df_results.to_csv(w_path + "/" + dgp + ".csv", index=False)

    # for i in range(N_SIM):
    # # for i in range(184, N_SIM):
    #     curr_result = one_simuation_iter(idx=i, theta=THETA, gamma=GAMMA, eta=ETA,
    #                                      sig_y=SIG_Y, pz=PZ, n_rep=N_REP,
    #                                      lin_y=LIN_Y, alpha=ALPHA)
    #     # with_header = i == 0
    #     # curr_result.to_csv("results/linear_dgp_noisy_network_N300.csv",
    #     #                    mode='a',
    #                        index=False)
    #                        # , header=with_header)
    #     print("Finished iteration {}".format(i))
        # df_sim_results = pd.concat([df_sim_results, curr_result])

    print("Elapsed time: ", time.time()-start)













