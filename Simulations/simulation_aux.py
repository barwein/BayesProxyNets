# Import necessary libraries

from src.MWG_sampler import MWG_sampler, MWG_init
from src.MCMC_fixed_net import mcmc_fixed_net
from src.MCMC_conti_relax import ContinuousRelaxationSampler
import src.Models as models
import src.GWG as gwg
import src.utils as utils
import pandas as pd
from jax import random


def one_simulation_iter(
    iter, idx, rng_key, data, new_interventions, cur_gamma_noise, true_vals, file_path
):
    results = []

    def run_eval(mcmc_obj, model_name):
        # 1. Dynamic
        dyn_stats = mcmc_obj.new_intervention_error_stats(
            new_z=new_interventions.Z_h,
            true_estimands=new_interventions.estimand_h,
            true_vals=true_vals,
        )
        results.append(
            {
                "idx": idx,
                "model": model_name,
                "estimand": "dynamic",
                "gamma_noise": cur_gamma_noise,
                **dyn_stats,
            }
        )

        # 2. Stochastic
        stoch_stats = mcmc_obj.new_intervention_error_stats(
            new_z=new_interventions.Z_stoch,
            true_estimands=new_interventions.estimand_stoch,
            true_vals=true_vals,
        )
        results.append(
            {
                "idx": idx,
                "model": model_name,
                "estimand": "stoch",
                "gamma_noise": cur_gamma_noise,
                **stoch_stats,
            }
        )

        # 3. GATE (Treat All vs None)
        gate_stats = mcmc_obj.new_intervention_error_stats(
            new_z=new_interventions.Z_gate,
            true_estimands=new_interventions.estimand_gate,
            true_vals=true_vals,
        )
        results.append(
            {
                "idx": idx,
                "model": model_name,
                "estimand": "gate",
                "gamma_noise": cur_gamma_noise,
                **gate_stats,
            }
        )

    print("--- fixed true net ---")
    rng_key = random.split(rng_key)[0]
    # --- fixed true network ---
    mcmc_true = mcmc_fixed_net(
        rng_key=rng_key,
        data=data,
        net_type="true",
        num_chains=4,
        progress_bar=False,
    )
    run_eval(mcmc_true, "true_net")

    print("--- fixed obs net ---")
    # --- observed network ---
    rng_key = random.split(rng_key)[0]

    mcmc_obs = mcmc_fixed_net(
        rng_key=rng_key,
        data=data,
        net_type="obs",
        num_chains=4,
        progress_bar=False,
    )
    run_eval(mcmc_obs, "obs_net")

    # --- MWG sampler (single proxy) ---
    print("--- MWG init params (single proxy) ---")

    rng_key = random.split(rng_key)[1]

    mwg_init = MWG_init(
        rng_key=rng_key,
        data=data,
    ).get_init_values()

    print("--- Sampling with MWG (single proxy) ---")

    rng_key = random.split(rng_key)[1]

    mwg_sampler = MWG_sampler(
        rng_key=rng_key,
        data=data,
        init_params=mwg_init,
        progress_bar=False,
        n_warmup=100,
        n_samples=100,
    )
    run_eval(mwg_sampler, "MWG")

    # --- MWG sampler without treatment model ---
    print("--- Sampling with MWG (no treatment model) ---")
    rng_key = random.split(rng_key)[1]
    mwg_sampler_no_z = MWG_sampler(
        rng_key=rng_key,
        data=data,
        init_params=mwg_init,
        gwg_fn=gwg.make_gwg_gibbs_fn_no_z,
        combined_model=models.combined_model,
        progress_bar=False,
        n_warmup=100,
        n_samples=100,
    )
    run_eval(mwg_sampler_no_z, "MWG_no_z")

    # --- MWG sampler (multiple proxies) ---
    print("--- MWG init params (multiple proxies) ---")

    rng_key = random.split(rng_key)[1]

    mwg_init_rep = MWG_init(
        rng_key=rng_key,
        data=data,
        cut_posterior_net_model=models.networks_marginalized_model_rep,
        triu_star_log_posterior_fn=models.compute_log_posterior_vmap_rep,
        progress_bar=False,
    ).get_init_values()

    # print("--- Sampling with MWG (multiple proxies) ---")
    rng_key = random.split(rng_key)[1]

    mwg_sampler_rep = MWG_sampler(
        rng_key=rng_key,
        data=data,
        init_params=mwg_init_rep,
        gwg_fn=gwg.make_gwg_gibbs_fn_rep,
        combined_model=models.combined_model_rep,
        progress_bar=False,
        n_warmup=100,
        n_samples=100,
    )
    run_eval(mwg_sampler_rep, "MWG_rep")

    # --- MWG sampler (multiple proxies) without treatment model ---
    # print("--- Sampling with MWG (multiple proxies, no treatment model) ---")
    # rng_key = random.split(rng_key)[1]
    # mwg_sampler_rep_no_z = MWG_sampler(
    #     rng_key=rng_key,
    #     data=data,
    #     init_params=mwg_init_rep,
    #     gwg_fn=gwg.make_gwg_gibbs_fn_rep_no_z,
    #     combined_model=models.combined_model_rep,
    #     progress_bar=False,
    #     # n_warmup=1000,
    #     # n_samples=1000,
    # )
    # run_eval(mwg_sampler_rep_no_z, "MWG_rep_no_z")

    # --- MWG sampler misspecified ---
    print("--- Sampling with MWG misspecified ---")
    rng_key = random.split(rng_key)[1]
    mwg_init_misspec = MWG_init(
        rng_key=rng_key,
        data=data,
        cut_posterior_net_model=models.networks_marginalized_model_misspec,
        triu_star_log_posterior_fn=models.compute_log_posterior_vmap_misspec,
        progress_bar=False,
        misspecified=True,
    ).get_init_values()

    rng_key = random.split(rng_key)[1]
    mwg_sampler_misspec = MWG_sampler(
        rng_key=rng_key,
        data=data,
        init_params=mwg_init_misspec,
        gwg_fn=gwg.make_gwg_gibbs_fn_misspec,
        combined_model=models.combined_model_misspec,
        progress_bar=False,
        misspecified=True,
        n_warmup=100,
        n_samples=100,
    )
    run_eval(mwg_sampler_misspec, "MWG_misspec")

    # --- MWG sampler misspecified (multiple proxies) ---
    # print("--- Sampling with MWG misspecified (multiple proxies) ---")
    # rng_key = random.split(rng_key)[1]
    # mwg_init_misspec_rep = MWG_init(
    #     rng_key=rng_key,
    #     data=data,
    #     cut_posterior_net_model=models.networks_marginalized_model_rep_misspec,
    #     triu_star_log_posterior_fn=models.compute_log_posterior_vmap_rep_misspec,
    #     progress_bar=False,
    #     misspecified=True,
    # ).get_init_values()
    # rng_key = random.split(rng_key)[1]
    # mwg_sampler_misspec_rep = MWG_sampler(
    #     rng_key=rng_key,
    #     data=data,
    #     init_params=mwg_init_misspec_rep,
    #     gwg_fn=gwg.make_gwg_gibbs_fn_rep_misspec,
    #     combined_model=models.combined_model_rep_misspec,
    #     progress_bar=False,
    #     misspecified=True,
    #     # n_warmup=1000,
    #     # n_samples=1000,
    # )
    # run_eval(mwg_sampler_misspec_rep, "MWG_misspec_rep")

    # --- Two-stage sampler ---
    print("--- Two-stage sampler ---")
    esti_triu_star = mwg_init["triu_star"][0, :]
    esti_exposures = utils.compute_exposures(esti_triu_star, data.Z)

    data_two_stage = data._replace(
        true_exposures=esti_exposures, triu_star=esti_triu_star
    )

    rng_key = random.split(rng_key)[1]
    # --- fixed estimated network (two-stage) ---
    mcmc_two_stage = mcmc_fixed_net(
        rng_key=rng_key,
        data=data_two_stage,
        net_type="true",
        num_chains=4,
        progress_bar=False,
    )
    run_eval(mcmc_two_stage, "two_stage")

    # --- Fixed estimated network (two-stage) from multiple proxies ---
    print("--- Two-stage sampler (multiple proxies) ---")
    esti_triu_star_rep = mwg_init_rep["triu_star"][0, :]
    esti_exposures_rep = utils.compute_exposures(esti_triu_star_rep, data.Z)

    data_two_stage_rep = data._replace(
        true_exposures=esti_exposures_rep, triu_star=esti_triu_star_rep
    )

    rng_key = random.split(rng_key)[1]
    mcmc_two_stage_rep = mcmc_fixed_net(
        rng_key=rng_key,
        data=data_two_stage_rep,
        net_type="true",
        num_chains=4,
        progress_bar=False,
    )
    run_eval(mcmc_two_stage_rep, "two_stage_rep")

    # =-- Continuous relaxation sampler ---
    print("--- Continuous relaxation sampler ---")
    rng_key = random.split(rng_key)[1]
    cont_relax_sampler = ContinuousRelaxationSampler(
        rng_key=rng_key,
        data=data,
        num_steps=20000,
        learning_rate=0.001,
        num_samples=5000,
        temperature=0.5,
        progress_bar=False,
    )
    cont_relax_sampler.run()
    run_eval(cont_relax_sampler, "cont_relax")

    # --- save results and write to csv ---
    results_df = pd.DataFrame(results)
    file_name = f"{file_path}/sim_results_{iter}.csv"
    header = iter == 0
    results_df.to_csv(file_name, index=False, mode="a", header=header)
