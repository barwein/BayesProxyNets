# Load libraries
import time
import numpy as np
import pandas as pd
import multiprocessing
# import os
import jax
# import jax.numpy as jnp
from jax import random
# from jax.scipy.special import expit
# import numpyro.distributions as dist
# import numpyro
# from numpyro.contrib.funsor import config_enumerate
# # from tqdm import tqdm
# from joblib import Parallel, delayed
# from numpyro.infer import MCMC, NUTS, Predictive

from src.Aux_functions import DataGeneration, Outcome_MCMC, Network_MCMC, Bayes_Modular, create_noisy_network


# parameters guides:
# theta: p(A* | X, theta)
# gamma: p(A | A*, X, gamma)
# eta, sig_y: p(Y | Z, X, A*, eta, sig_y)
# alpha: pi_alpha(z) ---> stochastic intervention

def one_simuation_iter(idx, theta, gamma, eta, sig_y, pz, n_rep, lin_y, alpha):
    rng_key = random.PRNGKey(idx)
    rng_key, rng_key_ = random.split(rng_key)

    # --- Get data ---
    df_oracle = DataGeneration(theta=theta, eta=eta, sig_y=sig_y, pz=pz, lin=lin_y, alpha=alpha).get_data()
    # Generate noisy network measurement
    obs_network = create_noisy_network(df_oracle["adj_mat"], gamma)
    # save observed df and update A* and triu
    df_obs = df_oracle.copy()
    df_obs["adj_mat"] = obs_network["obs_mat"]
    df_obs["triu"] = obs_network["triu_obs"]

    # --- network module ---
    network_mcmc = Network_MCMC(data=df_obs, rng_key=rng_key)
    # get posterior samples and predictive distributions
    network_post = network_mcmc.get_posterior_samples()
    network_mean_post = network_mcmc.mean_posterior()
    network_pred_samples = network_mcmc.predictive_samples()

    # --- Outcome module (linear & GP) ---
    # with true network
    oracle_outcome_mcmc = Outcome_MCMC(data=df_oracle, type="oracle", rng_key=rng_key, iter=i)
    oracle_results = oracle_outcome_mcmc.get_results()
    # with observed network
    obs_outcome_mcmc = Outcome_MCMC(data=df_obs, type="observed", rng_key=rng_key, iter=i)
    obs_results = obs_outcome_mcmc.get_results()

    #  --- cut-posterior ---
    # TODO: finish the implemenation of three, two, and one stage as written in `test' file.

    # Cut-posterior and plugin estimates
    # with 2S we need the `network_pred_mean_post`, and with the others the `network_pred` object
    # 2S
    # print("2S")
    cut_2S_mcmc = Bayes_Modular(data=df_obs, n=n, bm_type="cut-2S",
                                post_predictive=network_pred_mean_post, n_rep=n_rep, iter=i)
    cut_2S_mcmc.run_inference()
    cut_2S_results = cut_2S_mcmc.get_results()
    # print("3S")
    # 3S
    cut_3S_mcmc = Bayes_Modular(data=df_obs, n=n, bm_type="cut-3S",
                                post_predictive=network_pred, n_rep=n_rep, iter=i)
    cut_3S_mcmc.run_inference()
    cut_3S_results = cut_3S_mcmc.get_results()
    # print("plugin")
    # plugin
    plugin_mcmc = Bayes_Modular(data=df_obs, n=n, bm_type="plugin",
                                post_predictive=network_pred, iter=i)
    plugin_mcmc.run_inference()
    plugin_results = plugin_mcmc.get_results()

    # Combine all
    results_all = pd.concat([oracle_results, obs_results, cut_2S_results, cut_3S_results, plugin_results])
    results_all['iter'] = i
    return results_all


if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=14"
    # RANDOM_SEED = 892357143

    print("### Starting simulation ###")
    print("N_CORES: ", multiprocessing.cpu_count())
    print("N jax cpu devices: ", jax.local_device_count())

    # RANDOM_SEED = 5415020
    RANDOM_SEED = 6262523
    rng = np.random.default_rng(RANDOM_SEED)

    THETA = [-2, -0.5]
    GAMMA = [0.05, 0.25]
    ETA = [-1, -3, -0.5, 2.5]
    SIG_Y = .5
    PZ = 0.3
    BM_TYPES = ["cut-2S", "cut-3S", "plugin"]
    # N = 300
    N = 300
    N_SIM = 500
    N_REP = 1000

    start = time.time()
    df_sim_results = pd.DataFrame()


    # TODO: change the `for-loop' for jax.vmap
    # for i in range(N_SIM):
    for i in range(184, N_SIM):
        curr_result = one_simuation_iter(n=N, theta=THETA, gamma=GAMMA, eta=ETA, sig_y=SIG_Y, pz=PZ, i=i, n_rep=N_REP)
        with_header = i == 0
        curr_result.to_csv("linear_dgp_noisy_network_N300.csv",
                           mode='a',
                           index=False, header=with_header)
        df_sim_results = pd.concat([df_sim_results, curr_result])
        print("Finished iteration {}".format(i))

    print("Elapsed time: ", time.time()-start)
