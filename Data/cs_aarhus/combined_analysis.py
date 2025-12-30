import os

N_CORES = 4
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_CORES}"


import jax.numpy as jnp
from jax import random
import jax
import pandas as pd

import Data.cs_aarhus.util_data as ud
import Data.cs_aarhus.data_mcmc as dmcmc

#
print(f"Jax devices: {jax.devices()}")

# --- Global parameters ---

# ETA = jnp.array([-1, 3, 3])
ETA = jnp.array([-1, 3, 4])
RHO = 0.5
SIG_INV = 1.0


N_ITERATION = 300
# N_ITERATION = 1


FILE_NAME = "Data/cs_aarhus/cs_analysis_results.csv"
# FILE_NAME = "Data/cs_aarhus/cs_analysis_results_TEST.csv"

# --- aux function for one iteration ---


def one_iteration(rng_key, network_data, latent_layer, idx, with_header=False):
    # generate treatment and outcomes synthetic data
    rng_key, _ = random.split(rng_key)
    data_z_y = ud.generate_data(
        key=rng_key,
        triu_star=network_data["triu_star"],
        eta=ETA,
        rho=RHO,
        sig_inv=SIG_INV,
    )

    # full data dict
    data = data_z_y | network_data

    # Estimands
    rng_key, _ = random.split(rng_key)
    intervention_estimand = ud.get_intervention_estimand(
        key=rng_key, triu_star=network_data["triu_star"], eta=ETA
    )

    # --- Samplers ---
    results = []

    def run_eval(mcmc_obj, model_name):
        #  Stochastic
        stoch_stats = mcmc_obj.new_intervention_error_stats(
            new_z=intervention_estimand["Z_stoch"],
            true_estimands=intervention_estimand["estimand_stoch"],
            true_vals=true_vals,
        )
        results.append(
            {
                "idx": idx,
                "layer": latent_layer,
                "model": model_name,
                "estimand": "stoch",
                **stoch_stats,
            }
        )

        # 3. GATE aka TTE (Treat All vs None)
        gate_stats = mcmc_obj.new_intervention_error_stats(
            new_z=intervention_estimand["Z_gate"],
            true_estimands=intervention_estimand["estimand_gate"],
            true_vals=true_vals,
        )
        results.append(
            {
                "idx": idx,
                "layer": latent_layer,
                "model": model_name,
                "estimand": "gate",
                **gate_stats,
            }
        )

    true_vals = {
        "eta": ETA,
        "rho": RHO,
        "sig_inv": SIG_INV,
        "triu_star": network_data["triu_star"],
    }

    # True network

    print("--- Running for True network ---")

    rng_key, _ = random.split(rng_key)
    mcmc_true = dmcmc.mcmc_fixed_net(rng_key=rng_key, data=data, net_type="true")
    run_eval(mcmc_true, "true_net")

    # aggergate 'OR' network

    print("--- Running for Aggregate OR network ---")

    rng_key, _ = random.split(rng_key)
    mcmc_agg_or = dmcmc.mcmc_fixed_net(rng_key=rng_key, data=data, net_type="agg_or")
    run_eval(mcmc_agg_or, "agg_or")

    # aggregate 'AND' network

    print("--- Running for Aggregate AND network ---")

    rng_key, _ = random.split(rng_key)
    mcmc_agg_and = dmcmc.mcmc_fixed_net(rng_key=rng_key, data=data, net_type="agg_and")
    run_eval(mcmc_agg_and, "agg_and")

    # MWG init
    print("--- Init MWG values ---")

    rng_key, _ = random.split(rng_key)
    mwg_init = dmcmc.MWG_init(
        rng_key=rng_key,
        data=data,
        gwg_init_batch_len=int(1e4),
        gwg_init_steps=1,
    ).get_init_values()

    # Twostage sample
    print("--- Running TwoStage sampler ---")
    esti_triu_star = mwg_init["triu_star"][0, :]
    esti_exposures = ud.compute_exposures(esti_triu_star, data["Z"])

    data_ts = data.copy()
    data_ts["true_exposures"] = esti_exposures
    data_ts["triu_star"] = esti_triu_star

    rng_key, _ = random.split(rng_key)
    mcmc_ts = dmcmc.mcmc_fixed_net(
        rng_key=rng_key,
        data=data_ts,
        net_type="true",
    )
    run_eval(mcmc_ts, "two_stage")

    # MWG sampelr aka Block Gibbs
    print("--- Running MWG sampler ---")

    rng_key, _ = random.split(rng_key)
    mwg_sampler = dmcmc.MWG_sampler(
        rng_key=rng_key,
        data=data,
        init_params=mwg_init,
        progress_bar=False,
        gwg_batch_len=1,
        gwg_n_steps=1,
    )
    run_eval(mwg_sampler, "MWG")

    results_df = pd.DataFrame(results)
    results_df.to_csv(FILE_NAME, index=False, mode="a", header=with_header)


if __name__ == "__main__":
    # read global data
    cs_data = ud.network_data()
    layers = list(cs_data["adj_mat_dict"].keys())

    # run analysis
    with_header = True

    rng_key = random.PRNGKey(42)
    for layer in layers:
        # data for given latent layer
        obs_triu, latent_triu = ud.triu_array_obs_n_latent(cs_data["triu_dict"], layer)
        agg_or_triu = ud.aggregate_edges(obs_triu, "or")
        agg_and_triu = ud.aggregate_edges(obs_triu, "and")

        network_data = {
            "triu_vals": obs_triu,
            "triu_star": latent_triu,
            "agg_or_triu": agg_or_triu,
            "agg_and_triu": agg_and_triu,
        }

        for idx in range(N_ITERATION):
            print(f"--- Layer {layer};  iteration {idx} ---")

            rng_key, _ = random.split(rng_key)

            one_iteration(
                rng_key,
                network_data,
                layer,
                idx,
                with_header,
            )

            with_header = False
