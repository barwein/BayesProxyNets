
import numpy as np
import pandas as pd
import pyreadr
from jax import random
import data_wrangle as dw
import wrapper_functions as wrap

### Global variables and random key ###

rng_key = random.PRNGKey(0)
_, rng_key = random.split(rng_key)


NUM_REP_MS = 1500

### Run analysis ###

# Load data
full_df = pyreadr.read_r('37070-0001-Data.rda')
full_df = pd.DataFrame(full_df['da37070.0001'])

# Filter and clean data
cleaned_df = dw.clean_data(full_df)

#  keep only treated schools
cleaned_df = cleaned_df[cleaned_df['SCHTREAT_NUMERIC'] == 1]

# run network analysis by school
# all_data, all_stoch_trt_expos, all_post_obs_expos, all_post_stoch_expos = wrap.all_schools_network_run_and_posterior(cleaned_df)
all_data, all_stoch_trt_expos, all_post_obs_expos, all_post_stoch_expos = wrap.all_schools_network_run_and_posterior(cleaned_df[cleaned_df['SCHID'].isin([1.0,3.0,6.0])])

# run outcome regression with observed (ST) network
observed_network_results = wrap.observed_network_run(all_data, all_stoch_trt_expos, rng_key)

# onestage inference
onestage_results = wrap.onestage_run(all_data,
                                     all_stoch_trt_expos,
                                     all_post_obs_expos,
                                     all_post_stoch_expos,
                                     rng_key)

# multistage inference
multistage_results = wrap.multistage_run(all_data,
                                            all_stoch_trt_expos,
                                            all_post_obs_expos,
                                            all_post_stoch_expos,
                                            NUM_REP_MS,
                                            rng_key)

# save results
results_combined = pd.concat([observed_network_results, onestage_results, multistage_results])
results_combined.to_csv('palluck_et_al_analysis_results.csv', index=False)

# TODO: run analysis for all schools (test run with limited schools and samples)



