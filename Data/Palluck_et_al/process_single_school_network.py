import pickle
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
import pyreadr
import wrapper_functions as wrap
import data_wrangle as dw
import argparse
import lzma


# Load data
# full_df = pyreadr.read_r('37070-0001-Data.rda')
full_df = pyreadr.read_r(
    "/a/home/cc/math/barwein/Cluster_runs/BayesNetsProxy/data_analysis/37070-0001-Data.rda"
)
full_df = pd.DataFrame(full_df["da37070.0001"])

# Filter and clean data
cleaned_df = dw.clean_data(full_df)

#  keep only treated schools
cleaned_df = cleaned_df[cleaned_df["SCHTREAT_NUMERIC"] == 1]

# get unique school ids
school_ids = cleaned_df["SCHID"].unique()


def process_single_school(all_df, schid, output_dir):
    """
    Process a single school and save its results to a pickle file.

    Args:
        all_df: DataFrame containing all schools' data
        schid: School ID to process
        output_dir: Directory to save the results
    """
    # Record processing status
    # status_msg = f"Processing school ID: {schid}\n"
    # with open(os.path.join(output_dir, "processing_status.txt"), "a") as f:
    # f.write(status_msg)

    # Run the analysis for one school
    obs_data, stoch_trt_expos, post_obs_expos, post_stoch_expos = (
        wrap.one_school_iteration(all_df, schid)
    )

    # Save results for this school
    results = {
        "school_id": schid,
        "obs_data": obs_data,
        "stoch_trt_expos": np.array(stoch_trt_expos),
        "post_obs_expos": np.array(post_obs_expos),
        "post_stoch_expos": np.array(post_stoch_expos),
    }

    # output_file = os.path.join(output_dir, f"school_{schid}_results.pkl")
    output_file = os.path.join(output_dir, f"school_{schid}_results.xz")
    with lzma.open(output_file, "wb") as f:
        # with open(output_file, "wb") as f:
        # pickle.dump(results, f)
        pickle.dump(results, f)

    return output_file


# Save job_id
parser = argparse.ArgumentParser(description="Run simulation with a specific job ID")
parser.add_argument("job_id", type=int, help="Job ID to run the simulation")
args = parser.parse_args()
if args.job_id is None:
    raise ValueError("Error: No job id is given.")

cur_schid = school_ids[args.job_id]

# Process the school
output_dir = (
    "/a/home/cc/math/barwein/Cluster_runs/BayesNetsProxy/data_analysis/school_results/"
)
output_file = process_single_school(cleaned_df, cur_schid, output_dir)
