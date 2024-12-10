import pickle
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
import wrapper_functions as wrap
import data_wrangle as dw
from jax import random

# import pyreadr
import lzma

# Global variables
NUM_REP_MS = 2000
rng_key = random.PRNGKey(0)
_, rng_key = random.split(rng_key)


# Aux function to combine school network analysis results
def combine_school_results(output_dir):
    """
    Combine results from all processed schools.
    """
    obs_data_list = []
    stoch_trt_expos_list = []
    post_obs_expos_list = []
    post_stoch_expos_list = []

    # Load all pickle files in the directory
    for filename in sorted(os.listdir(output_dir)):  # sorting ensures consistent order
        # if filename.endswith("_results.pkl"):
        if filename.endswith("_results.xz"):
            # with open(os.path.join(output_dir, filename), "rb") as f:
            with lzma.open(os.path.join(output_dir, filename), "rb") as f:
                results = pickle.load(f)

            obs_data_list.append(results["obs_data"])
            stoch_trt_expos_list.append(results["stoch_trt_expos"])
            post_obs_expos_list.append(results["post_obs_expos"])
            post_stoch_expos_list.append(results["post_stoch_expos"])

    # Combine results
    all_data = dw.concatenate_dict_arrays(obs_data_list)
    all_stoch_trt_expos = jnp.concatenate(stoch_trt_expos_list, axis=-1)
    all_post_obs_expos = jnp.concatenate(post_obs_expos_list, axis=-1)
    all_post_stoch_expos = jnp.concatenate(post_stoch_expos_list, axis=-1)

    return all_data, all_stoch_trt_expos, all_post_obs_expos, all_post_stoch_expos


# combine results and run outcome models
output_dir = (
    "/a/home/cc/math/barwein/Cluster_runs/BayesNetsProxy/data_analysis/school_results/"
)

# Combine all school results
print("Combining school results...")
all_data, all_stoch_trt_expos, all_post_obs_expos, all_post_stoch_expos = (
    combine_school_results(output_dir)
)
all_data["school"] = dw.transform_schid(all_data["school"])

print("Running observed network analysis...")
# run outcome regression with observed (ST) network
observed_network_results = wrap.observed_network_run(
    all_data, all_stoch_trt_expos, rng_key
)

print("Running one-stage inference...")
# onestage inference
onestage_results = wrap.onestage_run(
    all_data, all_stoch_trt_expos, all_post_obs_expos, all_post_stoch_expos, rng_key
)

print("Running multi-stage inference...")
# multistage inference
multistage_results = wrap.multistage_run(
    all_data,
    all_stoch_trt_expos,
    all_post_obs_expos,
    all_post_stoch_expos,
    NUM_REP_MS,
    rng_key,
)

# save results
print("Saving final results...")
results_combined = pd.concat(
    [observed_network_results, onestage_results, multistage_results]
)
w_path = "/a/home/cc/math/barwein/Cluster_runs/BayesNetsProxy/data_analysis/"
res_file_name = os.path.join(w_path, "palluck_et_al_analysis_results.csv")
results_combined.to_csv(res_file_name, index=False)
