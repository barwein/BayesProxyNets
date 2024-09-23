import numpy as np
import pandas as pd
import os
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
from jax.scipy.special import expit, logit
import numpyro.distributions as dist
import numpyro
from numpyro.contrib.funsor import config_enumerate
# import
from numpyro.infer import MCMC, NUTS, Predictive, SVI, TraceEnum_ELBO, TraceGraph_ELBO
from numpyro.infer.autoguide import AutoNormal
import pyro
import torch
from hsgp.approximation import hsgp_squared_exponential, eigenfunctions
from hsgp.spectral_densities import diag_spectral_density_squared_exponential
import models_for_data_analysis as models
from tqdm import tqdm

from src.Aux_functions import N_CORES

# --- Global variables ---
N_CORES = 4
# N_CORES = 10
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_CORES}"
rng = np.random.default_rng(151)

# --- Utility functions ---
def n_edges_to_n_nodes(n_edges):
    """Compute the number of nodes from the number of edges"""
    return int((1 + jnp.sqrt(1 + 8*n_edges))/2)

# @jit
# def prop_treated_neighbors(trts, adj_mat):
#     """Compute the proportion of treated neighbors"""
#     n_treated = jnp.dot(adj_mat, trts)
#     degs = jnp.sum(adj_mat, axis=1)
#     non_zero_mask = (degs != 0)
#     prop_treated = jnp.where(non_zero_mask, n_treated / degs, 0.0)
#     return prop_treated

def Triu_to_mat(triu_v, n):
    """Convert a vector of the upper triangular part of a matrix to a symmetric matrix"""
    adj_mat = jnp.zeros((n,n))
    adj_mat = adj_mat.at[np.triu_indices(n=n, k=1)].set(triu_v)
    return adj_mat + adj_mat.T

@jit
def eigen_centrality(adj_mat):
    """Compute the eigenvector centrality"""
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = jnp.linalg.eigh(adj_mat)
    # Find the index of the largest eigenvalue
    largest_eigenvalue_index = jnp.argmax(eigenvalues)
    # Get the corresponding eigenvector
    largest_eigenvector = eigenvectors[:, largest_eigenvalue_index]
    # Scale the eigenvector
    norm = jnp.sign(largest_eigenvector.sum()) * jnp.linalg.norm(largest_eigenvector)
    scaled_eigenvector = largest_eigenvector / norm
    return scaled_eigenvector

@jit
def zeigen_value(Z, adj_mat):
    """Compute the Z-eigenvalue centrality"""
    eig_cen = eigen_centrality(adj_mat)
    if Z.ndim == 1:  # Case when Z has shape (N,)
        return jnp.dot(adj_mat, Z * eig_cen)
        # zeigen = jnp.dot(adj_mat, Z * eig_cen)
    elif Z.ndim == 2:  # Case when Z has shape (M, N)
        return jnp.dot(Z * eig_cen, adj_mat.T)  # Transpose A_mat for correct dimensions
        # zeigen = jnp.dot(Z * eig_cen, adj_mat.T)  # Transpose A_mat for correct dimensions
    # if subset is not None:
    #     return zeigen[subset]
    # else:
    #     return zeigen

@jit
def prop_treated_neighbors(Z, adj_mat):
    # degrees = jnp.sum(adj_mat, axis=1)
    if Z.ndim == 1:  # Case when Z has shape (N,)
        # Compute sum of treated neighbors
        treated_sum = jnp.dot(adj_mat, Z)
        # Compute proportion, avoiding division by zero
        # return jnp.where(degrees != 0, treated_sum / degrees, 0.0)
        return (treated_sum > 0).astype(jnp.int32)
    elif Z.ndim == 2:  # Case when Z has shape (M, N)
        # Compute sum of treated neighbors for each set of treatments
        treated_sum = jnp.dot(Z, adj_mat.T)  # Shape: (M, N)
        # Compute proportion, avoiding division by zero
        # return jnp.where(degrees != 0, treated_sum / degrees, 0.0)
        return (treated_sum > 0).astype(jnp.int32)

def stochastic_intervention(alpha, n, n_approx=500):
    z_stoch = rng.binomial(n=1, p=alpha, size=(n_approx, n))
    # z_stoch shape is (n_approx, n)
    return z_stoch

def trt_and_exposures_of_elig(new_trts_arr, new_exposures_arr, df):
    """Get the treatments and exposures of the eligible students"""
    elig_mask = df['ELIGIBLE'].values == 1
    elig_trts = new_trts_arr[:,:,elig_mask]
    elig_exposures = new_exposures_arr[:,:,elig_mask]
    return elig_trts, elig_exposures

# --- MCMC aux functions ---
def linear_model_samples_parallel(key, trts, exposures, sch_treat, fixed_df, grade, school, Y):
    """Run the parallel MCMC for the linear model"""
    kernel_outcome = NUTS(models.outcome_model)
    lin_mcmc = MCMC(kernel_outcome, num_warmup=2000, num_samples=2500,num_chains=4, progress_bar=False)
    lin_mcmc.run(key, trts=trts, exposures=exposures, sch_treat=sch_treat,
                 fixed_df=fixed_df, grade=grade, school=school, Y=Y)
    lin_mcmc.print_summary()
    return lin_mcmc.get_samples()

@jit
def linear_model_samples_vectorized(key, trts, exposures, sch_treat, fixed_df, grade, school, Y):
    """Run the vectorized MCMC for the linear model"""
    kernel_outcome = NUTS(models.outcome_model)
    lin_mcmc = MCMC(kernel_outcome, num_warmup=1000, num_samples=250, num_chains=1,
                    progress_bar=False, chain_method="vectorized")
    lin_mcmc.run(key, trts=trts, exposures=exposures, sch_treat=sch_treat,
                 fixed_df=fixed_df, grade=grade, school=school, Y=Y)
    return lin_mcmc.get_samples()

@jit
def outcome_jit_pred(trts, exposures, sch_treat, fixed_df, grade, school, post_samples, key):
    """Predictive function for the outcome model"""
    pred_func = Predictive(models.outcome_model, post_samples)
    return pred_func(key, trts=trts, exposures=exposures, sch_treat=sch_treat,
                     fixed_df=fixed_df, grade=grade, school=school)["Y"]

linear_model_pred = vmap(outcome_jit_pred, in_axes=(0, 0, None, None, None, None, None, None))

def linear_pred(trts, exposures, sch_treat, fixed_df, grade, school, post_samples, key):
    """Predictive function for the linear model"""
    if trts.ndim == 2:
        return jnp.mean(linear_model_pred(trts, exposures, sch_treat, fixed_df,
                                                  grade, school, post_samples, key), axis=0)
    if trts.ndim == 1:
        n_trts = trts.shape[0]
        return linear_model_pred(trts.reshape((1, n_trts)), exposures.reshape((1, n_trts)),
                                 sch_treat, fixed_df,
                                 grade, school, post_samples, key).squeeze()


# --- Network and outcome wrappers ---

class Network_SVI:
    """Class for training the network module and obtaining samples"""
    def __init__(self, x_df, triu_obs, n_iter=100, n_samples=100, network_model=models.one_noisy_networks_model):
    # def __init__(self, x_df, triu_obs, n_iter=20000, n_samples=10000, network_model=models.one_noisy_networks_model):
        self.x_df = x_df
        self.triu_obs = triu_obs
        self.N_edges = self.x_df.shape[0]
        self.N = n_edges_to_n_nodes(self.N_edges)
        # self.adj_mat = data["adj_mat"]
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.network_model = network_model
        self.guide = self.get_guide()

    def get_guide(self):
        return pyro.infer.autoguide.AutoLowRankMultivariateNormal(pyro.poutine.block(self.network_model,
                                                              hide=["triu_star"]),
                                      init_loc_fn = pyro.infer.autoguide.init_to_median())

    def train_model(self):
        pyro.clear_param_store()
        loss_func = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)
        optimzer = pyro.optim.ClippedAdam({"lr": 0.001})
        svi = pyro.infer.SVI(self.network_model, self.guide, optimzer, loss=loss_func)
        # losses_full = []
        for _ in range(self.n_iter):
        # for _ in tqdm(range(self.n_iter), desc="Training network model"):
            svi.step(self.x_df, self.triu_obs, self.N)
            # loss = svi.step(self.X_diff, self.X2_eq, self.triu, self.n)
            # losses_full.append(loss)

    def network_samples(self):
        triu_star_samples = []
        for _ in range(self.n_samples):
        # for _ in tqdm(range(self.n_samples), desc="Sampling A*"):
            # Get a trace from the guide
            guide_trace = pyro.poutine.trace(self.guide).get_trace(self.x_df, self.triu_obs, self.N)
            # Run infer_discrete
            inferred_model = pyro.infer.infer_discrete(pyro.poutine.replay(self.network_model, guide_trace),
                                            first_available_dim=-2)
            # Get a trace from the inferred model
            model_trace = pyro.poutine.trace(inferred_model).get_trace(self.x_df, self.triu_obs, self.N)
            # Extract triu_star from the trace
            triu_star_samples.append(model_trace.nodes['triu_star']['value'])
        # Convert to jnp array
        return jnp.stack(jnp.array(triu_star_samples))
        # triu_star_samples = torch.stack(triu_star_samples)
        # return jnp.array(triu_star_samples)

    def sample_triu_obs_predictive(self, num_samples=1000):
        """Sample triu_obs for posterior predictive checks"""
        predictive = pyro.infer.Predictive(self.network_model, guide=self.guide, num_samples=num_samples)

        # Generate samples
        with torch.no_grad():
            posterior_samples = predictive(self.x_df, None, self.N)

        # print(posterior_samples.keys())

        # Extract triu_obs samples
        if self.triu_obs.ndim == 1:
            triu_obs_samples = posterior_samples['obs_triu']
        else:
            triu_obs_A1 = posterior_samples['obs_triu_A1']
            triu_obs_A2 = posterior_samples['obs_triu_A2']
            triu_obs_samples = torch.stack([triu_obs_A1, triu_obs_A2])
        return triu_obs_samples


class Outcome_MCMC:
    """Class for training the outcome module and obtaining samples"""
    def __init__(self, data, rng_key):
        self.fixed_df = data["X"]
        self.school = data["school"]
        self.grade = data["grade"]
        self.trts = data["trts"]
        self.sch_trts = data["sch_trts"]
        self.exposures = data["exposures"]
        self.Y = data["Y"]
        # self.triu_v = data["triu_v"]
        # self.adj_mat = Triu_to_mat(self.triu_v, self.fixed_df.shape[0])
        # self.exposures = prop_treated_neighbors(self.trts, self.adj_mat)
        # self.exposures = zeigen_value(self.trts, self.adj_mat)
        self.rng_key = rng_key
        self.linear_post_samples = linear_model_samples_parallel(key=self.rng_key, trts=self.trts, exposures=self.exposures,
                                                                 sch_treat=self.sch_trts, fixed_df=self.fixed_df,
                                                                 grade=self.grade, school=self.school,
                                                                 Y=self.Y)

    def get_predicted_values(self, trts, exposures):
        if trts.ndim in [1, 2]:
            predictions = linear_pred(trts=trts, exposures=exposures, sch_treat=self.sch_trts,
                                      fixed_df=self.fixed_df, grade=self.grade, school=self.school,
                                      post_samples=self.linear_post_samples, key=self.rng_key)
            return predictions
        elif trts.ndim == 3:
            preds_list = []
            for i in range(trts.shape[0]):
                preds_list.append(linear_pred(trts=trts[i], exposures=exposures[i], sch_treat=self.sch_trts,
                                              fixed_df=self.fixed_df, grade=self.grade, school=self.school,
                                              post_samples=self.linear_post_samples, key=self.rng_key))
            return jnp.stack(preds_list)
        # return predictions
        else:
            raise ValueError("Invalid number of dimensions for trts")
            return None


# --- Multistage and onestage inference ---

# @jit
def posterior_exposures(triu_sample, trts, n):
    """Compute the posterior exposures values"""
    curr_Astar = Triu_to_mat(triu_sample, n=n)
    prop_trt = prop_treated_neighbors(trts, curr_Astar)
    # zeigen = zeigen_value(trts, curr_Astar)
    # return (prop_trt + zeigen)/2
    # return zeigen
    return prop_trt

parallel_post_exposures = pmap(posterior_exposures, in_axes=(0, None, None))
vectorized_post_exposures = vmap(posterior_exposures, in_axes=(0, None, None))

@jit
def one_linear_run(post_exposures, post_new_exposures, obs_trts, new_trts, sch_trts,
                   fixed_df, grade, school, Y, key) -> jnp.ndarray:
    """
    Run one iteration of the multistage linear model
    @param post_exposures: posterior exposures samples
    @param post_new_exposures: posterior new exposures samples, shape is (G,M,N) where G is the number of new treatments
    @param obs_trts: observed treatments
    @param new_trts: posterior new treatments samples, shape is (G,M,N) where G is the number of new treatments
    """
    # get samples from outcome model
    lin_samples = linear_model_samples_vectorized(key, obs_trts, post_exposures, sch_trts,
                                                  fixed_df, grade, school, Y)
    # get predictions
    num_new_trts = new_trts.shape[0]
    preds = []
    for i in range(num_new_trts):
        lin_pred = linear_pred(new_trts[i], post_new_exposures[i], sch_trts, fixed_df,
                               grade, school,
                               lin_samples, key)
        # lin_pred shape is (B, N)
        preds.append(lin_pred)

    res = jnp.stack(preds)
    # res shape is (G, B, N) where G = num_new_trts, B = num_post_samples, N = n
    # print(res.shape)
    return res
    # return jnp.array(preds)

parallel_linear_run = pmap(one_linear_run, in_axes=(0, 0, None, None, None, None, None, None, None, None))


def linear_multistage(post_exposures, post_new_exposures, obs_trts, new_trts, sch_trts, fixed_df,
                      grade, school, Y, key):
    num_net_samples = post_exposures.shape[0]
    n = obs_trts.shape[0]
    results = []
    for i in range(0, num_net_samples, N_CORES):
    # for i in tqdm(range(0, num_net_samples, N_CORES),desc="Multistage run"):
        i_results = parallel_linear_run(post_exposures[i:(i + N_CORES)],
                                        post_new_exposures[i:(i + N_CORES)],
                                        obs_trts, new_trts, sch_trts,
                                        fixed_df, grade, school,
                                        Y, key)
        results.append(i_results)
    results_arr = jnp.concatenate(results, axis=0)
    # print("result_c shape: ", results_arr.shape)
    result_arr = jnp.transpose(results_arr, axes=(1, 0, 2, 3))
    # print("result transposed shape: ", result_arr.shape)
    num_new_trts = new_trts.shape[0]
    num_post_samples = result_arr.shape[-2]
    result_arr = result_arr.reshape((num_new_trts, num_net_samples*num_post_samples , n))
    return result_arr
    # results_c = vectorized_multistage(multi_samp_nets, Y, Z_obs, Z_new, X, key)
    # n_samples = results_c.shape[2]
    # save error stats
    # results_lin_h = results_c[:, 0, :, :]
    # results_lin_h = results_c
    # results_lin_h_long = results_lin_h.reshape((B * n_samples, n))
    # return results_lin_h_long


class Onestage_MCMC:
    def __init__(self, obs_trts, post_exposures, sch_trts, fixed_df,
                 grade, school,Y, rng_key):
        self.obs_trts = obs_trts
        self.post_exposures = post_exposures
        self.sch_trts = sch_trts
        self.fixed_df = fixed_df
        self.grade = grade
        self.school = school
        self.Y = Y
        self.rng_key = rng_key
        self.linear_post_samples = linear_model_samples_parallel(key=self.rng_key, trts=self.obs_trts,
                                                                 exposures=self.post_exposures, sch_treat=self.sch_trts,
                                                                 fixed_df=self.fixed_df, grade=self.grade,
                                                                 school=self.school,
                                                                 Y=self.Y)

    def get_predicted_values(self, trts, exposures):
        if trts.ndim in [1, 2]:
            predictions = linear_pred(trts=trts, exposures=exposures, sch_treat=self.sch_trts,
                                      fixed_df=self.fixed_df, grade=self.grade, school=self.school,
                                      post_samples=self.linear_post_samples, key=self.rng_key)
            # prediction shape is (B, N) where B is the number of post samples and N is the number of nodes
            return predictions
        elif trts.ndim == 3:
            preds_list = []
            for i in range(trts.shape[0]):
                preds_list.append(linear_pred(trts=trts[i], exposures=exposures[i], sch_treat=self.sch_trts,
                                              fixed_df=self.fixed_df, grade=self.grade, school=self.school,
                                              post_samples=self.linear_post_samples, key=self.rng_key))
                #  prediction shape is (B, N) where B is the number of post samples and N is the number of nodes
            return jnp.stack(preds_list)
        # return predictions
        else:
            raise ValueError("Invalid number of dimensions for trts")
            return None