# Load libraries
import time
import numpy as np
import pandas as pd
# import multiprocessing
# import os
# import jax
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


def one_simuation_iter(n, theta, gamma, eta, sig_y, pz, iter):
    rng_key = random.PRNGKey(iter)
    rng_key, rng_key_ = random.split(rng_key)
    # Get data
    df_oracle = DataGeneration(n=n, theta=theta, eta=eta, sig_y=sig_y, pz=pz).get_data()
    # Generate noisy network measurement
    obs_network = create_noisy_network(df_oracle["adj_mat"], gamma, n)
    # save observed df and update A* and triu
    df_obs = df_oracle.copy()
    df_obs["adj_mat"] = obs_network["obs_mat"]
    df_obs["triu"] = obs_network["triu_obs"]

    print("Running outcomes models with A as given")
    # Run MCMC with true network (as given)
    oracle_outcome_mcmc = Outcome_MCMC(data=df_oracle, n=n, type="oracle", rng_key=rng_key, iter=iter)
    oracle_outcome_mcmc.run_outcome_model()
    oracle_results = oracle_outcome_mcmc.get_summary_outcome_model()
    # Run MCMC with observed network (as given)
    obs_outcome_mcmc = Outcome_MCMC(data=df_obs, n=n, type="observed", rng_key=rng_key, iter=iter)
    obs_outcome_mcmc.run_outcome_model()
    obs_results = obs_outcome_mcmc.get_summary_outcome_model()

    print("Running network model")
    # Run network module
    network_mcmc = Network_MCMC(data=df_obs, n=n, rng_key=rng_key)
    network_mcmc.run_network_model()
    # get predictive distributions
    network_pred = network_mcmc.get_network_predictive(mean_posterior=False)
    network_pred_mean_post = network_mcmc.get_network_predictive(mean_posterior=True)

    print("Running estimation with cut-posterior and plugin")
    # Cut-posterior and plugin estimates
    # with 2S we need the `network_pred_mean_post`, and with the others the `network_pred` object
    # 2S
    cut_2S_mcmc = Bayes_Modular(data=df_obs, n=n, bm_type="cut-2S",
                                post_predictive=network_pred_mean_post, n_rep=10, iter=iter)
    cut_2S_mcmc.run_inference()
    cut_2S_results = cut_2S_mcmc.get_results()
    # 3S
    cut_3S_mcmc = Bayes_Modular(data=df_obs, n=n, bm_type="cut-3S",
                                post_predictive=network_pred, n_rep=10, iter=iter)
    cut_3S_mcmc.run_inference()
    cut_3S_results = cut_3S_mcmc.get_results()
    # plugin
    plugin_mcmc = Bayes_Modular(data=df_obs, n=n, bm_type="plugin",
                                post_predictive=network_pred, iter=iter)
    plugin_mcmc.run_inference()
    plugin_results = plugin_mcmc.get_results()

    # Combine all
    results_all = pd.concat([oracle_results, obs_results, cut_2S_results, cut_3S_results, plugin_results])
    results_all['iter'] = iter
    return results_all


if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    RANDOM_SEED = 892357143
    rng = np.random.default_rng(RANDOM_SEED)

    THETA = [-2, -0.5]
    GAMMA = [0.05, 0.25]
    ETA = [-1, -3, 0.5, -0.25]
    SIG_Y = 1
    PZ = 0.3
    BM_TYPES = ["cut-2S", "cut-3S", "plugin"]
    N = 300
    N_SIM = 30

    start = time.time()
    twosims = pd.DataFrame()
    for iter in range(N_SIM):
        curr_result = one_simuation_iter(n=N, theta=THETA, gamma=GAMMA, eta=ETA,
                                         sig_y=SIG_Y, pz=PZ, iter=iter)
        with_header = iter == 0
        curr_result.to_csv("linear_dgp_noisy_network_N300.csv",
                           mode='a',
                           index=False, header=with_header)
        twosims = pd.concat([twosims,curr_result])
        print("Done iteration {}".format(iter))

    print("Elapsed time: ", time.time()-start)
