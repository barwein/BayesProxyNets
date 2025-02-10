from src.MWG_sampler import MWG_sampler, MWG_init
from src.MCMC_fixed_net import mcmc_fixed_net
import src.Models as models
import src.GWG as gwg
import pandas as pd
from jax import random


def one_simulation_iter(
    iter, idx, rng_key, data, new_interventions, cur_gamma_noise, file_path
):
    results = []

    print("--- fixed true net ---")
    rng_key = random.split(rng_key)[0]
    # --- fixed true network ---
    mcmc_true = mcmc_fixed_net(
        rng_key=rng_key,
        data=data,
        net_type="true",
        n_warmup=2000,
        n_samples=2500,
        num_chains=4,
        progress_bar=False,
    )

    true_net_dynamic_stats = mcmc_true.new_intervention_error_stats(
        new_z=new_interventions.Z_h, true_estimands=new_interventions.estimand_h
    )

    results.append(
        {
            "idx": idx,
            "model": "true_net",
            "estimand": "dynamic",
            "gamma_noise": cur_gamma_noise,
            **true_net_dynamic_stats,
        }
    )

    true_net_stoch_stats = mcmc_true.new_intervention_error_stats(
        new_z=new_interventions.Z_stoch, true_estimands=new_interventions.estimand_stoch
    )

    results.append(
        {
            "idx": idx,
            "model": "true_net",
            "estimand": "stoch",
            "gamma_noise": cur_gamma_noise,
            **true_net_stoch_stats,
        }
    )

    print("--- fixed obs net ---")
    # --- observed network ---
    rng_key = random.split(rng_key)[0]

    mcmc_obs = mcmc_fixed_net(
        rng_key=rng_key,
        data=data,
        net_type="obs",
        n_warmup=2000,
        n_samples=2500,
        num_chains=4,
        progress_bar=False,
    )

    obs_net_dynamic_stats = mcmc_obs.new_intervention_error_stats(
        new_z=new_interventions.Z_h, true_estimands=new_interventions.estimand_h
    )

    results.append(
        {
            "idx": idx,
            "model": "obs_net",
            "estimand": "dynamic",
            "gamma_noise": cur_gamma_noise,
            **obs_net_dynamic_stats,
        }
    )

    obs_net_stoch_stats = mcmc_obs.new_intervention_error_stats(
        new_z=new_interventions.Z_stoch, true_estimands=new_interventions.estimand_stoch
    )

    results.append(
        {
            "idx": idx,
            "model": "obs_net",
            "estimand": "stoch",
            "gamma_noise": cur_gamma_noise,
            **obs_net_stoch_stats,
        }
    )

    # --- MWG sampler (single proxy) ---
    print("--- MWG init params (single proxy) ---")

    rng_key = random.split(rng_key)[1]

    mwg_init = MWG_init(
        rng_key=rng_key,
        data=data,
        # n_warmup_networks=20,
        # n_samples_networks=20,
        # num_chains_networks=2,
        progress_bar=True,
    ).get_init_values()

    print("--- Sampling with MWG (single proxy) ---")

    rng_key = random.split(rng_key)[1]

    mwg_sampler = MWG_sampler(
        rng_key=rng_key,
        data=data,
        init_params=mwg_init,
        # n_warmup=2000,
        n_warmup=10,
        # n_samples=2500,
        n_samples=10,
        num_chains=4,
        progress_bar=True,
    )

    mwg_dynamic_stats = mwg_sampler.new_intervention_error_stats(
        new_z=new_interventions.Z_h, true_estimands=new_interventions.estimand_h
    )

    results.append(
        {
            "idx": idx,
            "model": "MWG",
            "estimand": "dynamic",
            "gamma_noise": cur_gamma_noise,
            **mwg_dynamic_stats,
        }
    )

    mwg_stoch_stats = mwg_sampler.new_intervention_error_stats(
        new_z=new_interventions.Z_stoch, true_estimands=new_interventions.estimand_stoch
    )

    results.append(
        {
            "idx": idx,
            "model": "MWG",
            "estimand": "stochastic",
            "gamma_noise": cur_gamma_noise,
            **mwg_stoch_stats,
        }
    )

    # --- MWG sampler (multiple proxies) ---
    print("--- MWG init params (multiple proxies) ---")

    rng_key = random.split(rng_key)[1]

    mwg_init_rep = MWG_init(
        rng_key=rng_key,
        data=data,
        cut_posterior_net_model=models.networks_marginalized_model_rep,
        triu_star_log_posterior_fn=models.compute_log_posterior_vmap_rep,
        # n_warmup_networks=20,
        # n_samples_networks=20,
        # num_chains_networks=2,
        progress_bar=True,
    ).get_init_values()

    # print("--- Sampling with MWG (multiple proxies) ---")
    rng_key = random.split(rng_key)[1]

    mwg_sampler_rep = MWG_sampler(
        rng_key=rng_key,
        data=data,
        init_params=mwg_init_rep,
        gwg_fn=gwg.make_gwg_gibbs_fn_rep,
        combined_model=models.combined_model_rep,
        n_warmup=10,
        n_samples=10,
        num_chains=4,
        # progress_bar=True,
    )

    mwg_dynamic_stats_rep = mwg_sampler_rep.new_intervention_error_stats(
        new_z=new_interventions.Z_h, true_estimands=new_interventions.estimand_h
    )

    results.append(
        {
            "idx": idx,
            "model": "MWG_rep",
            "estimand": "dynamic",
            "gamma_noise": cur_gamma_noise,
            **mwg_dynamic_stats_rep,
        }
    )

    mwg_stoch_stats_rep = mwg_sampler_rep.new_intervention_error_stats(
        new_z=new_interventions.Z_stoch, true_estimands=new_interventions.estimand_stoch
    )

    results.append(
        {
            "idx": idx,
            "model": "MWG_rep",
            "estimand": "stochastic",
            "gamma_noise": cur_gamma_noise,
            **mwg_stoch_stats_rep,
        }
    )

    # --- save results and write to csv ---
    results_df = pd.DataFrame(results)
    file_name = f"{file_path}/sim_results_{iter}.csv"
    header = iter == 0
    results_df.to_csv(file_name, index=False, mode="a", header=header)
