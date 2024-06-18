# Load libraries
import time
import numpy as np
import pandas as pd
import multiprocessing
import os
from itertools import combinations
from functools import partial
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
from jax.scipy.special import expit
import numpyro.distributions as dist
import numpyro
from numpyro.contrib.funsor import config_enumerate
# from tqdm import tqdm
from joblib import Parallel, delayed
from numpyro.infer import MCMC, NUTS, Predictive
from hsgp.approximation import hsgp_squared_exponential


# --- Set cores and seed ---
N_CORES = multiprocessing.cpu_count()
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_CORES}"
RANDOM_SEED = 892357143
rng = np.random.default_rng(RANDOM_SEED)

# --- Set global variables and values ---
N = 300
M = 20


# --- data generation for each iteration ---
class DataGeneration:
    def __init__(self, theta, eta, sig_y, pz, lin, alpha, n=N):
        self.n = n
        self.theta = theta
        self.eta = eta
        self.sig_y = sig_y
        self.lin = lin
        self.alpha = alpha
        self.X = self.generate_X()
        self.Z = self.generate_Z(p=pz)
        self.X_diff = self.x_diff()
        self.triu_dim = int(self.n*(self.n-1)/2)
        self.triu = self.generate_triu()
        self.adj_mat = self.generate_adj_matrix()
        self.eig_cen = eigen_centrality(self.adj_mat)
        self.zeigen = zeigen_value(self.Z, self.eig_cen, self.adj_mat)
        self.Y, self.epsi = self.gen_outcome(self.Z, self.zeigen)
        self.Z_h = self.dynamic_intervention()
        self.Z_stoch = self.stochastic_intervention()
        self.estimand_h = self.get_true_estimand(self.Z_h)
        self.estimand_stoch = self.get_true_estimand(self.Z_stoch)
        # self.exposures = self.generate_exposures()

    def generate_X(self, loc=0, scale=3):
        return rng.normal(loc=loc, scale=scale, size=self.n)

    def generate_Z(self, p=0.3):
        return rng.binomial(n=1, p=p, size=self.n)

    def x_diff(self):
        idx_pairs = combinations(range(self.n), 2)
        x_d = np.array([abs(self.X[i] - self.X[j]) for i, j in idx_pairs])
        return x_d

    def generate_triu(self):
        probs = expit(self.theta[0] + self.theta[1] * self.X_diff)
        return rng.binomial(n=1, p=probs, size=self.triu_dim)

    def generate_adj_matrix(self):
        mat = np.zeros((self.n, self.n))
        idx_upper_tri = np.triu_indices(n=self.n, k=1)
        mat[idx_upper_tri] = self.triu
        return mat + mat.T
    def gen_outcome(self, z, zeig):
        df_lin = np.transpose(np.array([[1]*self.n, z, self.X]))
        epsi = rng.normal(loc=0, scale=self.sig_y, size=N)
        if self.lin:
            mean_y = np.dot(np.column_stack((df_lin, zeig)), self.eta)
        else:
            mean_lin = np.dot(df_lin, self.eta[0:3])
            mean_nonlin = self.eta[3] / (1 + np.exp(-15 * (zeig - 0.4)))
            mean_y = mean_lin + mean_nonlin
        Y = mean_y + epsi
        return Y, epsi

    def dynamic_intervention(self, threshold=1.5):
        return np.where((self.X > threshold) | (self.X < -threshold), 1, 0)

    def stochastic_intervention(self, n_approx=1e3):
        return rng.binomial(n=1, p=self.alpha, size=(n_approx, self.n))

    def get_true_estimand(self, z_new):
        zeigen_new = zeigen_value(z_new, self.eig_cen, self.adj_mat)
        if z_new.ndim == 2:
            n_stoch = z_new.shape[0]
            results = np.zeros((n_stoch, 1))
            for i in range(n_stoch):
                df = np.transpose(np.array([[1] * self.n, z_new[i,], self.X]))
                y, epsi = self.gen_outcome(df, zeigen_new[i,])
                results[i,] = np.mean(y - epsi)
            return np.mean(results, axis=0).squeeze()
        else:
            # assert Z_stoch.ndim == 1
            df = np.transpose(jnp.array([[1] * self.n, z_new, self.X]))
            y, epsi = self.gen_outcome(df, zeigen_new)
            return np.mean(y - epsi)

    # def generate_exposures(self):
    #     return jnp.dot(self.adj_mat, self.Z)

    # def generate_outcome(self):
    #     mu_y = self.eta[0] + self.eta[1]*self.Z + self.eta[2]*self.exposures + self.eta[3]*self.X
    #     return mu_y + rng.normal(loc=0, scale=self.sig_y, size=self.n)

    def get_data(self):
        return {"Z" : self.Z, "X" : self.X, "Y" : self.Y,
                "triu" : self.triu, "Zeigen" : self.zeigen,
                "adj_mat" : self.adj_mat, "X_diff" : self.X_diff,
                "Z_h" : self.Z_h, "Z_stoch" : self.Z_stoch,
                "estimand_h" : self.estimand_h,
                "estimand_stoch" : self.estimand_stoch}


# --- General aux functions ---
def create_noisy_network(adj_mat, gamma):
    obs_mat = np.zeros((N, N))  # create nXn matrix of zeros
    triu_idx = np.triu_indices(n=N, k=1)
    obs_mat[triu_idx] = adj_mat[triu_idx]  # init as true network
    for i in range(0, N):  # add noise
        for j in range(i + 1, N):
            if adj_mat[i, j] == 1:
                obs_mat[i, j] = rng.binomial(n=1, p=1-gamma[1], size=1)[0]  # retain existing edge w.p. `1-gamma1`
            else:
                obs_mat[i, j] = rng.binomial(n=1, p=gamma[0], size=1)[0]  # add non-existing edge w.p. `gamma0`
    obs_mat = obs_mat + obs_mat.T
    triu_obs = obs_mat[triu_idx]
    return {"obs_mat" : obs_mat,
            "triu_obs" : triu_obs}

@jit
def Triu_to_mat(triu_v):
    adj_mat = jnp.zeros((N,N))
    # idx_utri = np.triu_indices(n=NN,k=1)
    adj_mat = adj_mat.at[np.triu_indices(n=N, k=1)].set(triu_v)
    return adj_mat + adj_mat.T

@jit
def eigen_centrality(adj_mat):
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
def zeigen_value(Z, eig_cen, adj_mat):
    if Z.ndim == 1:  # Case when Z has shape (N,)
        return jnp.dot(adj_mat, Z * eig_cen)
    elif Z.ndim == 2:  # Case when Z has shape (M, N)
        return jnp.dot(Z * eig_cen, adj_mat.T)  # Transpose A_mat for correct dimensions

def between_var(x, mean_all):
    n_rep = len(x)
    return (1/(n_rep - 1))*np.sum(np.square(x-mean_all))

def compute_error_stats(esti_post_draws, true_estimand, method="TEST", idx=None):
    mean = np.round(np.mean(esti_post_draws),3)
    medi = np.round(np.median(esti_post_draws),3)
    std = np.round(np.std(esti_post_draws),3)
    RMSE = np.round(np.sqrt(np.mean(np.power(esti_post_draws - true_estimand, 2))),3)
    q025 = np.quantile(esti_post_draws, 0.025)
    q975 = np.quantile(esti_post_draws, 0.975)
    cover = q025 <= true_estimand <= q975
    return pd.DataFrame([{"idx" : idx, "method" : method,
            "mean" : mean, "median" : medi, "true" : np.round(true_estimand,3),
            "bias" : np.round(mean - true_estimand,3), "std" : std, "RMSE" : RMSE,
            "q025" : np.round(q025,3), "q975" : np.round(q975,3), "covering" : cover}])


# --- NumPyro models ---
@config_enumerate
def network_model(X_diff, TriU):
    with numpyro.plate("theta_i", 2):
        theta = numpyro.sample("theta", dist.Normal(0, 10))
    mu_net = theta[0] + theta[1]*X_diff
    # triu_n = int(n * (n - 1) / 2)
    gamma0 = numpyro.sample("gamma0", dist.Uniform(low=0, high=0.5))
    gamma1 = numpyro.sample("gamma1", dist.Uniform(low=0, high=0.5))
    with numpyro.plate("A* and A", TriU.shape[0]):
        triu_star = numpyro.sample("triu_star", dist.Bernoulli(logits=mu_net),
                                   infer={"enumerate": "parallel"})
        prob_misspec = triu_star*(1 - gamma1) + (1 - triu_star)*gamma0
        numpyro.sample("obs_triu", dist.Bernoulli(probs=prob_misspec), obs=TriU)


def outcome_model(df, Y=None):
    with numpyro.plate("Lin coef.", df.shape[1]):
        eta = numpyro.sample("eta", dist.Normal(0, 10))
    sig = numpyro.sample("sig", dist.HalfNormal(scale=2))
    mu_y = jnp.dot(df, eta)
    with numpyro.plate("obs", df.shape[0]):
        numpyro.sample("Y", dist.Normal(loc=mu_y, scale=sig), obs=Y)

def HSGP_model(Xlin, Xgp, ell, m, Y=None, non_centered=True):
    # --- Priors ---
    magn = numpyro.sample("magn", dist.HalfNormal(2))
    length = numpyro.sample("length", dist.HalfNormal(5))
    sig = numpyro.sample("sig", dist.HalfNormal(2))
    # --- GP ---
    f = hsgp_squared_exponential(
        x=Xgp, alpha=magn, length=length, ell=ell, m=m, non_centered=non_centered)
    # --- Linear part ---
    with numpyro.plate("Lin coef.", Xlin.shape[1]):
        eta = numpyro.sample("eta", dist.Normal(0, 10))
    # --- Combine linear with GP ---
    mu = jnp.dot(Xlin, eta) + f
    # --- Likelihood ---
    with numpyro.plate("obs", Xlin.shape[0]):
        numpyro.sample("Y", dist.Normal(loc=mu, scale=sig), obs=Y)


# --- MCMC aux functions ---
@jit
def Astar_pred(key, post_samples, Xd, triu):
    pred_func = Predictive(model=network_model, posterior_samples=post_samples, infer_discrete=True,num_samples=1)
    return pred_func(key, X=Xd, TriU=triu)

# @jit
def linear_model_samples_parallel(key, Y, df):
    kernel_outcome = NUTS(outcome_model)
    lin_mcmc = MCMC(kernel_outcome, num_warmup=2000, num_samples=4000,num_chains=4, progress_bar=False)
    lin_mcmc.run(key, X=df, Y=Y)
    return lin_mcmc.get_samples()

@jit
def linear_model_samples_vectorized(key, Y, df):
    kernel_outcome = NUTS(outcome_model)
    lin_mcmc = MCMC(kernel_outcome, num_warmup=400, num_samples=150, num_chains=1,
                    progress_bar=False, chain_method="vectorized")
    lin_mcmc.run(key, X=df, Y=Y)
    return lin_mcmc.get_samples()

@jit
def outcome_jit_pred(post_samples, df_arr, key):
    pred_func = Predictive(outcome_model, post_samples)
    return pred_func(key, df_arr)

# @jit
def HSGP_model_samples_parallel(key, Y, Xgp, Xlin, ell):
    kernel_hsgp = NUTS(HSGP_model)
    hsgp_mcmc = MCMC(kernel_hsgp, num_warmup=2000, num_samples=4000,num_chains=4, progress_bar=False)
    hsgp_mcmc.run(key, Xgp=Xgp, Xlin=Xlin, ell=ell ,m=M, Y=Y)
    return hsgp_mcmc.get_samples()

@jit
def HSGP_model_samples_vectorized(key, Y, Xgp, Xlin, ell):
    kernel_hsgp = NUTS(HSGP_model)
    hsgp_mcmc = MCMC(kernel_hsgp, num_warmup=400, num_samples=150,num_chains=1,
                     progress_bar=False, chain_method="vectorized")
    hsgp_mcmc.run(key, Xgp=Xgp, Xlin=Xlin, ell=ell ,m=M, Y=Y)
    return hsgp_mcmc.get_samples()

@jit
def HSGP_jit_pred(post_samples, Xgp, Xlin, ell, key):
    pred_func = Predictive(HSGP_model, post_samples)
    return pred_func(key, Xgp=Xgp, Xlin = Xlin, ell=ell, m=M)

@jit
def linear_model_outcome_pred(z, zeigen, post_samples, x):
    df = jnp.transpose(jnp.array([[1] * x.shape[0], z, x, zeigen]))
    pred = outcome_jit_pred(post_samples, df)
    return jnp.mean(pred["Y"], axis=1)

linear_model_pred = vmap(linear_model_outcome_pred, in_axes=(0, 0, None, None))

def linear_pred(z, zeigen, post_samples, x):
    if z.ndim == 2:
        return linear_model_pred(z, zeigen, post_samples, x)
    if z.ndim == 1:
        n_z = z.shape[0]
        return linear_model_pred(z.reshape((1, n_z)), zeigen.reshape((1, n_z)), post_samples, x)

@jit
def hsgp_model_outcome_pred(z, zeigen, post_samples, x, ell):
    # ell_ = jnp.array(c*jnp.max(jnp.abs(zeigen))).reshape(1,1)
    df = jnp.transpose(jnp.array([[1] * x.shape[0], z, x, zeigen]))
    pred = HSGP_jit_pred(post_samples, Xgp=df[:, 3:], Xlin=df[:, 0:3], ell=ell)
    return jnp.mean(pred["Y"], axis=1)

hsgp_model_pred = vmap(hsgp_model_outcome_pred, in_axes=(0, 0, None, None, None))

def hsgp_pred(z, zeigen, post_samples, x, ell):
    if z.ndim == 2:
        return hsgp_model_pred(z, zeigen, post_samples, x, ell)
    if z.ndim == 1:
        n_z = z.shape[0]
        return hsgp_model_pred(z.reshape((1, n_z)), zeigen.reshape((1, n_z)), post_samples, x, ell)


@jit
def get_mcmc_samples(key, Y, Z, X, A):
    kernel_outcome = NUTS(outcome_model)
    mcmc = MCMC(kernel_outcome, num_warmup=500, num_samples=250,
                num_chains=2, progress_bar=False)
    mcmc.run(key, Y=Y, Z=Z, X=X, A=A, n=N)
    return mcmc.get_samples()



class Network_MCMC:
    def __init__(self, data, n, rng_key, n_warmup=1000, n_samples=3000, n_chains=4):
    # def __init__(self, data, n, rng_key, n_warmup=1000, n_samples=2000, n_chains=6):
        self.X_diff = data["X_diff"]
        self.triu = data["triu"]
        self.adj_mat = data["adj_mat"]
        self.n = n
        self.rng_key = rng_key
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.network_m = self.network()

    def network(self):
        kernel = NUTS(network_model)
        return MCMC(kernel, num_warmup=self.n_warmup, num_samples=self.n_samples,
                    num_chains=self.n_chains, progress_bar=False)

    def run_network_model(self):
        self.network_m.run(self.rng_key, X_diff=self.X_diff, TriU=self.triu, n=self.n)

    def get_network_predictive(self, mean_posterior = False):
        posterior_samples = self.network_m.get_samples()
        if mean_posterior:
            posterior_mean = {"theta": np.expand_dims(np.mean(posterior_samples["theta"], axis=0), -2),
                              "gamma0": np.expand_dims(np.mean(posterior_samples["gamma0"]), -1),
                              "gamma1": np.expand_dims(np.mean(posterior_samples["gamma1"]), -1)}
            return Predictive(model=network_model, posterior_samples=posterior_mean,
                              infer_discrete=True, num_samples=1)
        else:
            posterior_predictive = Predictive(model=network_model, posterior_samples=posterior_samples,
                                              infer_discrete=True)
            return posterior_predictive(self.rng_key, X_diff=self.X_diff, TriU=self.triu, n=self.n)["triu_star"]


class Outcome_MCMC:
    def __init__(self, data, n, type, rng_key, iter, suff_stat=False, n_warmup=1000, n_samples=3000, n_chains=4):
        self.X = data["X"]
        self.Z = data["Z"]
        self.Y = data["Y"]
        self.exposures = data["exposures"]
        self.adj_mat = data["adj_mat"]
        self.n = n
        self.type = type
        self.rng_key = rng_key
        self.iter = iter
        self.suff_stat = suff_stat
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.outcome_m = self.outcome()

    def outcome(self):
        if not self.suff_stat:
            kernel = NUTS(outcome_model)
        else:
            kernel = NUTS(outcome_stat_model)

        return MCMC(kernel, num_warmup=self.n_warmup, num_samples=self.n_samples,
                    num_chains=self.n_chains, progress_bar=False)

    def run_outcome_model(self):
        if not self.suff_stat:
            self.outcome_m.run(self.rng_key, Y=self.Y,Z=self.Z,X=self.X,A=self.adj_mat,n=self.n)
        else:
            self.outcome_m.run(self.rng_key, Y=self.Y,Z=self.Z,X=self.X,expos=self.exposures,n=self.n)

    def get_summary_outcome_model(self):
        posterior_samples = self.outcome_m.get_samples()
        mean_posterior = np.mean(posterior_samples["eta"],axis=0)[2]
        median_posterior = np.median(posterior_samples["eta"],axis=0)[2]
        std_posterior = np.std(posterior_samples["eta"],axis=0)[2]
        q005_posterior = np.quantile(posterior_samples["eta"],q=0.005,axis=0)[2]
        q025_posterior = np.quantile(posterior_samples["eta"],q=0.025,axis=0)[2]
        q975_posterior = np.quantile(posterior_samples["eta"],q=0.975,axis=0)[2]
        q995_posterior = np.quantile(posterior_samples["eta"],q=0.995,axis=0)[2]
        min_posterior = np.min(posterior_samples["eta"],axis=0)[2]
        max_posterior = np.max(posterior_samples["eta"],axis=0)[2]
        return pd.DataFrame({'mean' : mean_posterior,
                             'median' : median_posterior,
                             'std' : std_posterior,
                             'q005' : q005_posterior,
                             'q025' : q025_posterior,
                             'q975' : q975_posterior,
                             'q995' : q995_posterior,
                             'min' : min_posterior,
                             'max' : max_posterior,
                             'type' : self.type},
                            index = [self.iter])


class Bayes_Modular:
    def __init__(self, data, n, bm_type, post_predictive, n_rep=1000, n_warmup=500,
                 n_samples=250, n_chains=2, iter=0, n_cores=N_CORES):
        self.X = data["X"]
        self.Z = data["Z"]
        self.Y = data["Y"]
        self.exposures = data["exposures"]
        self.X_diff = data["X_diff"]
        self.triu = data["triu"]
        self.adj_mat = data['adj_mat']
        self.n = n
        self.type = bm_type # one of ["cut-2S", "cut-3S", "plugin"]
        self.post_predictive = post_predictive
        self.n_rep = n_rep
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.iter = iter
        self.n_cores = n_cores
        # self.Y_mcmc = self.MCMC_obj()
        self.results = None

    # def MCMC_obj(self):
    #     return MCMC(NUTS(outcome_model), num_warmup=self.n_warmup, num_samples=self.n_samples,
    #                 num_chains=2, progress_bar=False)

    def stage_aux(self, i):
        # sample network
        if self.type == "cut-2S":
            curr_mat = self.post_predictive(random.PRNGKey(i), X_diff=self.X_diff,
                                            TriU=self.triu, n=self.n)
            curr_mat = triu_to_mat(curr_mat["triu_star"], self.n)
        else: # self.type == "cut-3S":
            curr_mat = triu_to_mat(self.post_predictive[i,], self.n)
        # Run MCMC
        # self.Y_mcmc.run(random.PRNGKey(i**2), Y=self.Y, Z=self.Z, X=self.X, A=curr_mat, n=self.n)
        # curr_posterior_samples = self.Y_mcmc.get_samples()
        curr_posterior_samples = get_mcmc_samples(key=random.PRNGKey(i), Y=self.Y, Z=self.Z, X=self.X, A=curr_mat)
        # save results
        alpha_shape = curr_posterior_samples["eta"].shape
        converted_post_samp = {"iter": i, "sig": curr_posterior_samples["sig"]}
        for j in range(alpha_shape[1]):
            converted_post_samp["eta" + '_' + str(j)] = curr_posterior_samples["eta"][:, j]
        return pd.DataFrame(converted_post_samp)

    def twostage_inferene(self):
        # twostage_post_samp = Parallel(n_jobs=self.n_cores)(delayed(self.stage_aux)(i) for i in range(self.n_rep))
        # return pd.concat(twostage_post_samp, axis=0)
        twostage_post_samp = pd.DataFrame()
        for i in range(self.n_rep):
            cur_res = self.stage_aux(i)
            twostage_post_samp = pd.concat([twostage_post_samp, cur_res])
        return twostage_post_samp

    def threestage_inference(self):
        i_range = np.random.choice(a=range(self.post_predictive.shape[0]),
                                   size=self.n_rep, replace=False)
        # threestage_post_samp = Parallel(n_jobs=self.n_cores)(delayed(self.stage_aux)(i) for i in i_range)
        # return pd.concat(threestage_post_samp, axis=0)
        threestage_post_samp = pd.DataFrame()
        for i in i_range:
            cur_res = self.stage_aux(i)
            threestage_post_samp = pd.concat([threestage_post_samp, cur_res])
        return threestage_post_samp

    def plugin_aux(self, i_range):
        # deg_list = []
        expos_list = []
        for i in i_range:
            # sample network
            curr_mat = triu_to_mat(self.post_predictive[i,], self.n)
            # save statistics
            # deg_list.append([np.sum(curr_mat, 1)])
            expos_list.append([jnp.dot(curr_mat, self.Z)])
        return {'expos': expos_list}
        # return {'deg': deg_list, 'expos': expos_list}

    def plugin_inference(self):
        return self.plugin_aux(range(self.post_predictive.shape[0]))

    def run_inference(self):
        if self.type == "cut-3S":
            self.results = self.threestage_inference()
        if self.type == "cut-2S":
            self.results = self.twostage_inferene()
        if self.type == "plugin":
            self.results = self.plugin_inference()

    def cut_posterior_summarized(self):
        # agg_results = self.results.agg(['mean','median',q025, q975,'min','max'])
        agg_results = self.results.agg(['mean','median',q005, q025, q975, q995,'min','max'])
        mean_eta2 = agg_results["eta_2"]["mean"]
        # eta2_agg_by_iter = agg_results[["iter", "eta_2"]].groupby("iter").agg(["mean", "var"])
        eta2_agg_by_iter = self.results[["iter", "eta_2"]].groupby("iter").agg(["mean", "var"])
        eta2_agg_by_iter.columns = ["mean", "var"]
        # Get var-between and -within
        eta2_VB = between_var(eta2_agg_by_iter["mean"], mean_eta2)
        eta2_VW = np.mean(eta2_agg_by_iter["var"])
        eta2_std_MI = np.sqrt(eta2_VB*(1 + 1/self.n_rep) + eta2_VW)
        # Save all
        eta2_results_dict = agg_results["eta_2"].to_dict()
        eta2_results_dict["std"] = eta2_std_MI
        eta2_results_dict["type"] = self.type
        # eta2_results_dict = {k : eta2_results_dict[k] for k in ["mean","median","std","q025","q975","min","max","type"]}
        eta2_results_dict = {k : eta2_results_dict[k] for k in ["mean","median","std","q005","q025","q975",
                                                                "q995","min","max","type"]}
        return pd.DataFrame(eta2_results_dict, index=[self.iter])

    def plugin_posterior_summarized(self):
        self.exposures = np.mean(self.results["expos"], axis=0)
        cur_data = {'X' : self.X, 'Z' : self.Z, 'Y' : self.Y,
                    'exposures' : self.exposures, 'adj_mat' : self.adj_mat}
        plugin_outcome_m = Outcome_MCMC(data=cur_data,
                                        n=self.n,
                                        type=self.type,
                                        rng_key=random.split(random.PRNGKey(self.iter))[0],
                                        iter=self.iter,
                                        suff_stat=True)
        plugin_outcome_m.run_outcome_model()
        return plugin_outcome_m.get_summary_outcome_model()

    def get_results(self):
        if self.type in ["cut-2S", "cut-3S"]:
            return self.cut_posterior_summarized()
        if self.type in ["plugin"]:
            return self.plugin_posterior_summarized()
