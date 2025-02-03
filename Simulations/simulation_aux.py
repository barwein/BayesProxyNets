from src.MWG_sampler import MWG_sampler, MWG_init
from src.MCMC_fixed_net import mcmc_fixed_net
import pandas as pd


def one_simulation_iter(
    idx, rng_key, data, new_interventions, cur_gamma_noise, file_path
):
    results = []

    print("@@@ running with fixed true net @@@")
    # fixed true network
    mcmc_true = mcmc_fixed_net(
        rng_key=rng_key,
        data=data,
        net_type="true",
        n_warmup=2000,
        n_samples=2500,
        num_chains=2,
        progress_bar=True,
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

    # observed network
    print("@@@ running with fixed obs net @@@")

    mcmc_obs = mcmc_fixed_net(
        rng_key=rng_key,
        data=data,
        net_type="obs",
        n_warmup=2000,
        n_samples=2500,
        num_chains=2,
        progress_bar=True,
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

    # MWG sampler
    print("@@@ Getting MWG init params @@@")

    mwg_init = MWG_init(
        rng_key=rng_key, data=data, num_chains=2, progress_bar=True
    ).get_init_values()

    print("@@@ Sampling with MWG @@@")

    mwg_sampler = MWG_sampler(
        rng_key=rng_key,
        data=data,
        init_params=mwg_init,
        n_warmup=2000,
        n_samples=2500,
        num_chains=2,
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

    # save results and write to csv
    results_df = pd.DataFrame(results)
    file_name = f"{file_path}/results_{idx}.csv"
    results_df.to_csv(file_name, index=False, mode="a")
