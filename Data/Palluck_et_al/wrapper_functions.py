import numpy as np
import pandas as pd
import jax.numpy as jnp
import torch
from pandas import DataFrame

import utils_for_inference as util
import data_wrangle as dw
import models_for_data_analysis as models


### Global variables ###
ALPHAS = [0.3, 0.5, 0.7]
# ALPHAS = [0.3, 0.7]
MODELS_NAMES = ['one_noisy', 'repeated_noisy', 'multilayer']

### Functions ###
def one_school_network_analysis(df: pd.DataFrame):
    """
    Run network analysis for one school.
    :param df: data frame of the specific school
    :return: jnp.array of all 3 observed networks, jnp.array of posterior samples for each model
    """
    # Get observed networks
    ST_net = dw.network_by_school(df,  dw.ST_COLS, False)
    ST_W2_net = dw.network_by_school(df, dw.ST_W2_COLS, False)
    BF_net = dw.network_by_school(df, dw.BF_COLS, False)

    # Get covariates for network analysis
    cov_for_net = dw.create_net_covar_df(df)

    # Run analysis for each network model
    # 1. One noisy network model
    one_noisy_net = util.Network_SVI(x_df = cov_for_net,
                                      triu_obs = dw.adj_to_triu(ST_net),
                                      network_model=models.one_noisy_networks_model,
                                      n_iter = 20000,
                                      n_samples = 10000)
    one_noisy_net.train_model()
    post_one_noisy_net = one_noisy_net.network_samples()

    # 2. Two noisy networks model
    two_noisy_net = util.Network_SVI(x_df = cov_for_net,
                                     triu_obs = torch.stack([dw.adj_to_triu(ST_net),
                                                             dw.adj_to_triu(ST_W2_net)]),
                                     network_model=models.repeated_noisy_networks_model,
                                     n_iter = 20000,
                                     n_samples = 10000)
    two_noisy_net.train_model()
    post_two_noisy_net = two_noisy_net.network_samples()

    # 3. Multilayer networks model
    multilayer_net = util.Network_SVI(x_df = cov_for_net,
                                      triu_obs = torch.stack([dw.adj_to_triu(ST_net),
                                                              dw.adj_to_triu(BF_net)]),
                                      network_model=models.multilayer_networks_model,
                                      n_iter = 20000,
                                      n_samples = 10000)
    multilayer_net.train_model()
    post_multilayer_net = multilayer_net.network_samples()

    # return networks and posterior samples
    obs_nets_array = jnp.array([ST_net, ST_W2_net, BF_net])
    post_samples_array = jnp.array([post_one_noisy_net, post_two_noisy_net, post_multilayer_net])
    return obs_nets_array, post_samples_array


def get_stoch_treatments_and_obs_exposures(df, adj_mat):
    new_stoch_trts = []
    new_stoch_obs_exposures = []
    #  samples stochastic treatments and compute observed exposures
    for i in range(len(ALPHAS)):
        new_stoch_trts.append(util.stochastic_intervention(alpha=ALPHAS[i], n=df.shape[0]))
        new_stoch_obs_exposures.append(util.zeigen_value(Z=new_stoch_trts[i], adj_mat=adj_mat))

    new_stoch_trts = jnp.stack(new_stoch_trts)
    new_stoch_obs_exposures = jnp.stack(new_stoch_obs_exposures)

    all_trt_exposures = jnp.stack([new_stoch_trts, new_stoch_obs_exposures])

    # get subset for eligible units only
    stoch_trt_elig, stoch_exposures_elig = util.trt_and_exposures_of_elig(new_stoch_trts,
                                                                          new_stoch_obs_exposures,
                                                                          df)
    elig_trt_exposures = jnp.stack([stoch_trt_elig, stoch_exposures_elig])

    return all_trt_exposures, elig_trt_exposures

def posterior_stoch_exposures(post_net_samples_array, all_stoch_trts, df):
    n_units = df.shape[0]
    results = []
    for i in range(post_net_samples_array.shape[0]):
        cur_net_post_expos = []
        for j in range(all_stoch_trts.shape[1]):
            cur_post_expos = util.vectorized_post_exposures(post_net_samples_array[i],
                                                            all_stoch_trts[0, j],
                                                            n_units)
            cur_net_post_expos.append(cur_post_expos)
        results.append(jnp.stack(cur_net_post_expos))
    all_results = jnp.stack(results)
    elig_mask = df['ELIGIBLE'].values == 1
    return all_results[:,:,:,:,elig_mask]

def posterior_exposure_for_obs_treatments(post_net_samples_array, obs_treatments, df):
    n_units = df.shape[0]
    results = []
    for i in range(post_net_samples_array.shape[0]):
        cur_net_post_expos = []
        cur_post_expos = util.vectorized_post_exposures(post_net_samples_array[i],
                                                        obs_treatments,
                                                        n_units)
        results.append(cur_post_expos)
    all_results = jnp.stack(results)
    elig_mask = df['ELIGIBLE'].values == 1
    return all_results[:,:,elig_mask]

def one_school_iteration(all_df, schid):
    # get school df
    df_school = all_df[all_df['SCHID'] == schid]
    # Run network analysis
    all_nets, net_post_samples = one_school_network_analysis(df_school)
    # get stochastic interventions + observed exposures
    all_stoch_expos, elig_stoch_expos = get_stoch_treatments_and_obs_exposures(df_school, all_nets[0])
    # Get posterior exposures for observed treatments for each network model
    post_obs_exposure = posterior_exposure_for_obs_treatments(net_post_samples,
                                                              jnp.array(df_school['TREAT_NUMERIC'].values),
                                                              df_school)
    # Posterior exposures for stochastic interventions
    post_stoch_expos = posterior_stoch_exposures(net_post_samples, all_stoch_expos, df_school)
    # observed data for outcome regression
    obs_data = dw.data_for_outcome_regression(df_school, all_nets[0])

    return obs_data, elig_stoch_expos, post_obs_exposure, post_stoch_expos


def all_schools_network_run_and_posterior(all_df):
    # get unique school ids
    school_ids = all_df['SCHID'].unique()
    obs_data_list = []
    stoch_trt_expos_list = []
    post_obs_expos_list = []
    post_stoch_expos_list = []
    # run for each school
    for schid in school_ids:
        # print("running for schid: ", schid)
        obs_data, stoch_trt_expos, post_obs_expos, post_stoch_expos = one_school_iteration(all_df, schid)
        obs_data_list.append(obs_data)
        stoch_trt_expos_list.append(stoch_trt_expos)
        post_obs_expos_list.append(post_obs_expos)
        post_stoch_expos_list.append(post_stoch_expos)

    all_data = dw.concatenate_dict_arrays(obs_data_list)
    all_stoch_trt_expos = jnp.concatenate(stoch_trt_expos_list, axis=-1)
    all_post_obs_expos = jnp.concatenate(post_obs_expos_list, axis=-1)
    all_post_stoch_expos = jnp.concatenate(post_stoch_expos_list, axis=-1)

    return all_data, all_stoch_trt_expos, all_post_obs_expos, all_post_stoch_expos


def compute_summary_of_predictions(predictions, estimand_name, model_name, method_type):
    """
    :param predictions: array with shape (M, N) where M is number of posterior samples and N is number of units
    :param estimand: name of the estimand (e.g., 'stoch_30')
    :param model_name: name of the network model used for posterior draws ('observed', 'one_noisy', etc.)
    :param method_type: one of ('observed', 'onestage', 'multistage')
    :return: pd.Dataframe of results
    """
    # Get the mean across units (aka estimand)
    estimand_mean = jnp.mean(predictions, axis=1) # shape (M,)
    # Get mean and median of estimand over posterior samples
    post_mean = jnp.mean(estimand_mean).item()
    post_median = jnp.median(estimand_mean).item()
    # Get the posterior standard deviation of the estimand
    post_std = jnp.std(estimand_mean).item()
    # Get the 95% credible interval
    q025 = jnp.quantile(estimand_mean, 0.025).item()
    q975 = jnp.quantile(estimand_mean, 0.975).item()
    # Create a DataFrame with the results
    results_df = pd.DataFrame({'post_mean': [post_mean],
                               'post_median': [post_median],
                               'post_std': [post_std],
                               'q025': [q025],
                               'q975': [q975],
                               'estimand': [estimand_name],
                               'model': [model_name],
                               'method': [method_type]})
    return results_df


def observed_network_run(all_data, all_stoch_trt_expos, key):
    # run the model
    outcome_model = util.Outcome_MCMC(data=all_data, rng_key=key)
    # get predicted values
    pred_values = outcome_model.get_predicted_values(trts=all_stoch_trt_expos[0],
                                                      exposures=all_stoch_trt_expos[1])
    # save results for each new treatment
    results = []
    for i in range(len(ALPHAS)):
        trt_name = 'stoch_' + str(int(ALPHAS[i]*100))
        results.append(compute_summary_of_predictions(pred_values[i],
                                                    trt_name,
                                                    'observed',
                                                    'observed'))
    return pd.concat(results)


def onestage_run(all_data, all_stoch_trt_expos, all_post_obs_expos, all_post_stoch_expos, key):
    """
    wrap onestage inference for all models and obtaining predicted values for all new treatments
    :param all_data:
    :param all_stoch_trt_expos:
    :param all_post_obs_expos:
    :param all_post_stoch_expos:
    :param key:
    :return:
    """

    assert len(ALPHAS) == all_stoch_trt_expos.shape[0]

    results_list = []
    for i in range(len(MODELS_NAMES)):
        # Run one-stage estimation
        onestage_results = util.Onestage_MCMC(obs_trts=all_data['trts'],
                                                post_exposures= all_post_obs_expos[i].mean(axis=0),
                                                sch_trts = all_data['sch_trts'],
                                                fixed_df=all_data['X'],
                                                grade = all_data['grade'],
                                                school = all_data['school'],
                                                Y = all_data['Y'],
                                                rng_key=key)
        # get predictions
        pred = onestage_results.get_predicted_values(trts=all_stoch_trt_expos[0],
                                                  exposures = all_post_stoch_expos[i].mean(axis=1))
        # save results for each new treatment
        for j in range(len(ALPHAS)):
            trt_name = 'stoch_' + str(int(ALPHAS[j]*100))
            results_df = compute_summary_of_predictions(pred[j],
                                                        trt_name,
                                                        MODELS_NAMES[i],
                                                        'onestage')
            results_list.append(results_df)
        # Get the results in a DataFrame
    return pd.concat(results_list)

def multistage_run(all_data, all_stoch_trt_expos, all_post_obs_expos, all_post_stoch_expos, num_rep, key):
    """
    wrap multistage inference for all models and obtaining predicted values for all new treatments
    :param all_data:
    :param all_stoch_trt_expos:
    :param all_post_obs_expos:
    :param all_post_stoch_expos:
    :param num_rep: number of multistage iterations
    :param key:
    :return:
    """

    assert len(ALPHAS) == all_stoch_trt_expos.shape[0]
    reshaped_post_stoch_expos = jnp.transpose(all_post_stoch_expos, axes=(0,2,1,3,4))
    i_range = np.random.choice(a=range(all_post_obs_expos.shape[1]),
                               size=num_rep,
                               replace=False)
    results_list = []
    for i in range(len(MODELS_NAMES)):
        # Run multi-stage estimation
        multistage_preds = util.linear_multistage(post_exposures = all_post_obs_expos[i][i_range],
                                            post_new_exposures = reshaped_post_stoch_expos[i][i_range],
                                            obs_trts = all_data['trts'],
                                            new_trts = all_stoch_trt_expos[0],
                                            sch_trts = all_data['sch_trts'],
                                            fixed_df = all_data['X'],
                                            grade = all_data['grade'],
                                            school = all_data['school'],
                                            Y = all_data['Y'],
                                            key=key)
        # save results for each new treatment
        for j in range(len(ALPHAS)):
            trt_name = 'stoch_' + str(int(ALPHAS[j]*100))
            results_df = compute_summary_of_predictions(multistage_preds[j],
                                                        trt_name,
                                                        MODELS_NAMES[i],
                                                        'multistage')
            results_list.append(results_df)
        # Get the results in a DataFrame
    return pd.concat(results_list)