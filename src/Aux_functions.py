# Load libraries
import time
import numpy as np
import pandas as pd
import os
from itertools import combinations
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
from jax.scipy.special import expit
import numpyro.distributions as dist
import numpyro
from numpyro.contrib.funsor import config_enumerate
# import
from numpyro.infer import MCMC, NUTS, Predictive
from hsgp.approximation import hsgp_squared_exponential


# --- Set cores and seed ---
N_CORES = 4
# N_CORES = 20
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_CORES}"
RANDOM_SEED = 892357143
rng = np.random.default_rng(RANDOM_SEED)

# --- Set global variables and values ---
N = 300
M = 20
C = 3.5

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
        # return rng.normal(loc=loc, scale=scale, size=self.n)
        return jnp.array(rng.normal(loc=loc, scale=scale, size=self.n))

    def generate_Z(self, p=0.3):
        # return rng.binomial(n=1, p=p, size=self.n)
        return jnp.array(rng.binomial(n=1, p=p, size=self.n))

    def x_diff(self):
        idx_pairs = combinations(range(self.n), 2)
        # x_d = np.array([abs(self.X[i] - self.X[j]) for i, j in idx_pairs])
        x_d = jnp.array([abs(self.X[i] - self.X[j]) for i, j in idx_pairs])
        return x_d

    def generate_triu(self):
        probs = expit(self.theta[0] + self.theta[1] * self.X_diff)
        return jnp.array(rng.binomial(n=1, p=probs, size=self.triu_dim))
        # return rng.binomial(n=1, p=probs, size=self.triu_dim)

    def generate_adj_matrix(self):
        mat = np.zeros((self.n, self.n))
        idx_upper_tri = np.triu_indices(n=self.n, k=1)
        mat[idx_upper_tri] = self.triu
        return mat + mat.T

    def gen_outcome(self, z, zeig):
        df_lin = jnp.transpose(np.array([[1]*self.n, z, self.X]))
        epsi = jnp.array(rng.normal(loc=0, scale=self.sig_y, size=self.n))
        if self.lin:
            mean_y = jnp.dot(jnp.column_stack((df_lin, zeig)), self.eta)
            # mean_y = np.dot(np.column_stack((df_lin, zeig)), self.eta)
        else:
            mean_lin = jnp.dot(df_lin, self.eta[0:3])
            mean_nonlin = self.eta[3] / (1 + jnp.exp(-15 * (zeig - 0.4)))
            mean_y = mean_lin + mean_nonlin
        Y = jnp.array(mean_y + epsi)
        return Y, epsi

    def dynamic_intervention(self, threshold=1.5):
        return jnp.where((self.X > threshold) | (self.X < -threshold), 1, 0)
        # return jnp.where((self.X > threshold) | (self.X < -threshold), 1, 0)

    def stochastic_intervention(self, n_approx=100):
    # def stochastic_intervention(self, n_approx=1000):
        return rng.binomial(n=1, p=self.alpha, size=(n_approx, self.n))

    def get_true_estimand(self, z_new):
        zeigen_new = zeigen_value(z_new, self.eig_cen, self.adj_mat)
        if z_new.ndim == 2:
            n_stoch = z_new.shape[0]
            results = np.zeros((n_stoch, 1))
            for i in range(n_stoch):
                # df = np.transpose(np.array([[1] * self.n, z_new[i,], self.X]))
                y, epsi = self.gen_outcome(z_new[i,], zeigen_new[i,])
                results[i,] = jnp.mean(y - epsi)
            return jnp.mean(results, axis=0).squeeze()
        else:
            # assert Z_stoch.ndim == 1
            # df = np.transpose(np.array([[1] * self.n, z_new, self.X]))
            y, epsi = self.gen_outcome(z_new, zeigen_new)
            return jnp.mean(y - epsi)

    def get_data(self):
        return {"Z" : self.Z, "X" : self.X, "Y" : self.Y,
                "triu" : self.triu, "Zeigen" : self.zeigen,
                "eig_cen" : self.eig_cen,
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

# def compute_error_stats(esti_post_draws, true_estimand, method="TEST", estimand = "h", idx=None):
def compute_error_stats(esti_post_draws, true_estimand, idx=None):
    mean = jnp.round(jnp.mean(esti_post_draws),3)
    medi = jnp.round(jnp.median(esti_post_draws),3)
    std = jnp.round(jnp.std(esti_post_draws),3)
    RMSE = jnp.round(jnp.sqrt(jnp.mean(jnp.power(esti_post_draws - true_estimand, 2))),3)
    q025 = jnp.quantile(esti_post_draws, 0.025)
    q975 = jnp.quantile(esti_post_draws, 0.975)
    cover = (q025 <= true_estimand) & (true_estimand <= q975)
    # cover = q025 <= true_estimand <= q975
    return jnp.array([idx, mean, medi, jnp.round(true_estimand,3),
                      jnp.round(mean - true_estimand,3), std, RMSE,
                      jnp.round(q025,3), jnp.round(q975,3), cover])
 # return pd.DataFrame([{"idx" : idx, "method" : method, "estimand" : estimand,
 #            "mean" : mean, "median" : medi, "true" : jnp.round(true_estimand,3),
 #            "bias" : jnp.round(mean - true_estimand,3), "std" : std, "RMSE" : RMSE,
 #            "q025" : jnp.round(q025,3), "q975" : jnp.round(q975,3), "covering" : cover}])


# --- NumPyro models ---
@config_enumerate
def network_model(X_diff, TriU=None):
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
    # --- priors ---
    with numpyro.plate("Lin coef.", df.shape[1]):
        eta = numpyro.sample("eta", dist.Normal(0, 10))
    sig = numpyro.sample("sig", dist.HalfNormal(scale=2))
    mu_y = jnp.dot(df, eta)
    # --- likelihood --
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
# @jit
def linear_model_samples_parallel(key, Y, df):
    kernel_outcome = NUTS(outcome_model)
    lin_mcmc = MCMC(kernel_outcome, num_warmup=500, num_samples=1000,num_chains=4, progress_bar=False)
    # lin_mcmc = MCMC(kernel_outcome, num_warmup=2000, num_samples=4000,num_chains=4, progress_bar=False)
    lin_mcmc.run(key, df=df, Y=Y)
    return lin_mcmc.get_samples()

@jit
def linear_model_samples_vectorized(key, Y, df):
    kernel_outcome = NUTS(outcome_model)
    lin_mcmc = MCMC(kernel_outcome, num_warmup=200, num_samples=50, num_chains=1,
    # lin_mcmc = MCMC(kernel_outcome, num_warmup=400, num_samples=150, num_chains=1,
                    progress_bar=False, chain_method="vectorized")
    lin_mcmc.run(key, df=df, Y=Y)
    return lin_mcmc.get_samples()

@jit
def outcome_jit_pred(post_samples, df_arr, key):
    pred_func = Predictive(outcome_model, post_samples)
    return pred_func(key, df_arr)

# @jit
def HSGP_model_samples_parallel(key, Y, Xgp, Xlin, ell):
    kernel_hsgp = NUTS(HSGP_model)
    hsgp_mcmc = MCMC(kernel_hsgp, num_warmup=500, num_samples=1000,num_chains=4, progress_bar=False)
    # hsgp_mcmc = MCMC(kernel_hsgp, num_warmup=2000, num_samples=4000,num_chains=4, progress_bar=False)
    hsgp_mcmc.run(key, Xgp=Xgp, Xlin=Xlin, ell=ell ,m=M, Y=Y)
    return hsgp_mcmc.get_samples()

@jit
def HSGP_model_samples_vectorized(key, Y, Xgp, Xlin, ell):
    kernel_hsgp = NUTS(HSGP_model)
    hsgp_mcmc = MCMC(kernel_hsgp, num_warmup=200, num_samples=50,num_chains=1,
    # hsgp_mcmc = MCMC(kernel_hsgp, num_warmup=400, num_samples=150,num_chains=1,
                     progress_bar=False, chain_method="vectorized")
    hsgp_mcmc.run(key, Xgp=Xgp, Xlin=Xlin, ell=ell ,m=M, Y=Y)
    return hsgp_mcmc.get_samples()

@jit
def HSGP_jit_pred(post_samples, Xgp, Xlin, ell, key):
    pred_func = Predictive(HSGP_model, post_samples)
    return pred_func(key, Xgp=Xgp, Xlin = Xlin, ell=ell, m=M)

@jit
def linear_model_outcome_pred(z, zeigen, post_samples, x, key):
    df = jnp.transpose(jnp.array([[1] * N, z, x, zeigen]))
    pred = outcome_jit_pred(post_samples, df, key)
    return jnp.mean(pred["Y"], axis=1)

linear_model_pred = vmap(linear_model_outcome_pred, in_axes=(0, 0, None, None, None))

def linear_pred(z, zeigen, post_samples, x, key):
    if z.ndim == 2:
        return linear_model_pred(z, zeigen, post_samples, x, key)
    if z.ndim == 1:
        n_z = z.shape[0]
        return linear_model_pred(z.reshape((1, n_z)), zeigen.reshape((1, n_z)), post_samples, x, key)

@jit
def hsgp_model_outcome_pred(z, zeigen, post_samples, x, ell, key):
    # ell_ = jnp.array(c*jnp.max(jnp.abs(zeigen))).reshape(1,1)
    df = jnp.transpose(jnp.array([[1] * x.shape[0], z, x, zeigen]))
    pred = HSGP_jit_pred(post_samples, Xgp=df[:, 3:], Xlin=df[:, 0:3], ell=ell, key=key)
    return jnp.mean(pred["Y"], axis=1)

hsgp_model_pred = vmap(hsgp_model_outcome_pred, in_axes=(0, 0, None, None, None, None))

def hsgp_pred(z, zeigen, post_samples, x, ell, key):
    if z.ndim == 2:
        return hsgp_model_pred(z, zeigen, post_samples, x, ell, key)
    if z.ndim == 1:
        n_z = z.shape[0]
        return hsgp_model_pred(z.reshape((1, n_z)), zeigen.reshape((1, n_z)), post_samples, x, ell, key)


@jit
def Astar_pred(i, post_samples, Xd, triu):
    pred_func = Predictive(model=network_model, posterior_samples=post_samples, infer_discrete=True, num_samples=1)
    sampled_net = pred_func(random.PRNGKey(i), X_diff=Xd, TriU=triu)["triu_star"]
    return jnp.squeeze(sampled_net, axis=0)

vectorized_astar_pred = vmap(Astar_pred, in_axes=(0, None, None, None))
parallel_astar_pred = pmap(Astar_pred, in_axes=(0, None, None, None))


def get_many_post_astars(K, post_pred_mean, x_diff, triu_v):
    i_range = jnp.arange(K)
    return vectorized_astar_pred(i_range, post_pred_mean, x_diff, triu_v)


class Network_MCMC:
    def __init__(self, data, rng_key, n_warmup=250, n_samples=500, n_chains=4):
    # def __init__(self, data, rng_key, n_warmup=2000, n_samples=4000, n_chains=4):
    # def __init__(self, data, n, rng_key, n_warmup=1000, n_samples=2000, n_chains=6):
        self.X_diff = data["X_diff"]
        self.triu = data["triu"]
        self.adj_mat = data["adj_mat"]
        self.rng_key = rng_key
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.network_m = self.network()
        self.post_samples = None

    def network(self):
        kernel = NUTS(network_model)
        return MCMC(kernel, num_warmup=self.n_warmup, num_samples=self.n_samples,
                    num_chains=self.n_chains, progress_bar=False)

    def get_posterior_samples(self):
        self.network_m.run(self.rng_key, X_diff=self.X_diff, TriU=self.triu)
        self.post_samples = self.network_m.get_samples()
        return self.post_samples

    def mean_posterior(self):
        return {"theta": jnp.expand_dims(jnp.mean(self.post_samples["theta"], axis=0), -2),
                              "gamma0": jnp.expand_dims(np.mean(self.post_samples["gamma0"]), -1),
                              "gamma1": jnp.expand_dims(jnp.mean(self.post_samples["gamma1"]), -1)}

    def predictive_samples(self):
        posterior_predictive = Predictive(model=network_model, posterior_samples=self.post_samples,
                                          infer_discrete=True)
        return posterior_predictive(self.rng_key, X_diff=self.X_diff, TriU=self.triu)["triu_star"]


class Outcome_MCMC:
    def __init__(self, data, type, rng_key, iter):
        self.X = data["X"]
        self.Z = data["Z"]
        self.Y = jnp.array(data["Y"])
        self.adj_mat = data["adj_mat"]
        self.eig_cen = eigen_centrality(self.adj_mat)
        self.zeigen = zeigen_value(self.Z, self.eig_cen, self.adj_mat)
        self.n = N
        self.ell = jnp.array(C*jnp.max(jnp.abs(self.zeigen))).reshape(1,1)
        self.df = self.get_df()
        self.Z_h = data["Z_h"]
        self.Z_stoch = data["Z_stoch"]
        self.estimand_h = data["estimand_h"]
        self.estimand_stoch = data["estimand_stoch"]
        self.type = type
        self.rng_key = rng_key
        self.iter = iter
        self.linear_post_samples = linear_model_samples_parallel(key=self.rng_key, Y=self.Y, df=self.df)
        self.hsgp_post_samples = HSGP_model_samples_parallel(key=self.rng_key, Y=self.Y, Xgp=self.zeigen,
                                                             Xlin=self.df[:,0:3], ell=self.ell)

    def get_df(self):
        return jnp.transpose(jnp.array([[1]*self.n, self.Z, self.X, self.zeigen]))

    def get_results(self):
        # dynamic (h) intervention
        h_zeigen = zeigen_value(self.Z_h, self.eig_cen, self.adj_mat)
        linear_h_pred = linear_pred(self.Z_h, h_zeigen, self.linear_post_samples,
                                    self.X, self.rng_key)
        linear_h_stats = compute_error_stats(jnp.mean(linear_h_pred, axis=0),
                                                  self.estimand_h, idx=self.iter)
        hsgp_h_pred = hsgp_pred(self.Z_h, h_zeigen, self.hsgp_post_samples,
                                self.X, self.ell, self.rng_key)
        hsgp_h_stats = compute_error_stats(jnp.mean(hsgp_h_pred, axis=0),
                                                  self.estimand_h, idx=self.iter)
        # stochastic intervention
        stoch_zeigen = zeigen_value(self.Z_stoch, self.eig_cen, self.adj_mat)
        linear_stoch_pred = linear_pred(self.Z_stoch, stoch_zeigen,
                                        self.linear_post_samples, self.X, self.rng_key)
        linear_stoch_stats = compute_error_stats(jnp.mean(linear_stoch_pred, axis=0),
                                             self.estimand_h, idx=self.iter)
        hsgp_stoch_pred = hsgp_pred(self.Z_stoch, stoch_zeigen, self.hsgp_post_samples,
                                    self.X, self.ell, self.rng_key)
        hsgp_stoch_stats = compute_error_stats(jnp.mean(hsgp_stoch_pred, axis=0),
                                           self.estimand_h, idx=self.iter)

        return jnp.vstack([linear_h_stats, hsgp_h_stats, linear_stoch_stats, hsgp_stoch_stats])
        # return pd.concat([linear_h_stats, hsgp_h_stats, linear_stoch_stats, hsgp_stoch_stats])

# --- Modular bayesian inference from cut-posterior ---
# Aux functions



@jit
def compute_net_stats(Astar, Z):
    cur_eigen_cen = eigen_centrality(Astar)
    cur_Zeigen = zeigen_value(Z, cur_eigen_cen, Astar)
    cur_ell = C*jnp.max(jnp.abs(cur_Zeigen))
    cur_ell = jnp.array(cur_ell).reshape(1, 1)
    return cur_Zeigen, cur_ell

@jit
def get_samples_new_Astar(key, Y, Z, X, curr_Astar):
    cur_Zeigen, ell = compute_net_stats(curr_Astar, Z)
    # get df
    cur_df = jnp.transpose(jnp.array([[1] * N, Z, X, cur_Zeigen]))
    # Run MCMC
    cur_lin_samples = linear_model_samples_vectorized(key, Y, cur_df)
    cur_hsgp_samples = HSGP_model_samples_vectorized(key, Y=Y, Xgp=cur_df[:, 3:],
                                           Xlin=cur_df[:, 0:3], ell=ell)
    return cur_lin_samples, cur_hsgp_samples, ell

@jit
def get_predicted_values(key, z, zeigen, x, lin_samples, hsgp_samples, ell):
    # get predicted values
    # each has shape (#lin_samples, n)
    cur_lin_pred = linear_pred(z, zeigen, lin_samples, x, key)
    cur_hsgp_pred = hsgp_pred(z, zeigen, hsgp_samples, x, ell, key)
    # get estimands for each sample (sample mean across units)
    lin_estimates = jnp.mean(cur_lin_pred, axis=0)
    hsgp_estimates = jnp.mean(cur_hsgp_pred, axis=0)
    return lin_estimates, hsgp_estimates


@jit
def multistage_mcmc(samp_net, Y, Z_obs, Z_new, X, key):
    # sample network
    curr_Astar = Triu_to_mat(samp_net)
    # re-run MCMC with new network
    curr_lin_samples, curr_hsgp_samples, cur_ell = get_samples_new_Astar(key, Y, Z_obs, X, curr_Astar)
    zeigen_new, _ = compute_net_stats(curr_Astar, Z_new)
    # get predicted estimands for new `Z' values
    lin_estimates, hsgp_estimates = get_predicted_values(key, Z_new, zeigen_new, X, curr_lin_samples,
                                                         curr_hsgp_samples, cur_ell)
    return jnp.array([lin_estimates, hsgp_estimates])

parallel_multistage = pmap(multistage_mcmc, in_axes=(0, None, None, None, None, None))


def multistage_run(multi_samp_nets, Y, Z_obs, Z_new, X, K, type, z_type, iter, true_estimand, key):
    results = []
    for i in range(0,K, N_CORES):
        i_results = parallel_multistage(multi_samp_nets[i:(i+N_CORES),], Y,
                                        Z_obs, Z_new, X, key)
        results.append(i_results)
    results_c = jnp.concatenate(results, axis=0)
    n_samples = results_c.shape[2]
    results_lin = results_c[:,0,:]
    results_lin_long = results_lin.reshape(K*n_samples)
    results_gp = results_c[:,1,:]
    results_gp_long = results_gp.reshape(K*n_samples)
    error_stats_lin = compute_error_stats(esti_post_draws=results_lin_long,
                                          true_estimand=true_estimand,
                                          idx=iter)
    error_stats_gp = compute_error_stats(esti_post_draws=results_gp_long,
                                          true_estimand=true_estimand,
                                          idx=iter)
    return jnp.vstack([error_stats_lin, error_stats_gp])
    # return pd.concat([error_stats_lin, error_stats_gp])


@jit
def network_posterior_stats(triu_sample, z):
    curr_Astar = Triu_to_mat(triu_sample)
    cur_eig_cen = eigen_centrality(curr_Astar)
    zeigen = zeigen_value(z, cur_eig_cen, curr_Astar)
    return zeigen

parallel_network_stats = pmap(network_posterior_stats, in_axes=(0, None))

def get_onestage_stats(multi_triu_samples, Z_obs, Z_h, Z_stoch):
    obs_zeigen = []
    h_zeigen = []
    stoch_zeigen = []
    for i in range(0, multi_triu_samples.shape[0], N_CORES):
        obs_results = parallel_network_stats(multi_triu_samples[i:(i + N_CORES), ], Z_obs)
        obs_zeigen.append(obs_results)
        new_results = parallel_network_stats(multi_triu_samples[i:(i + N_CORES), ], Z_h)
        h_zeigen.append(new_results)
        stoch_results = parallel_network_stats(multi_triu_samples[i:(i + N_CORES), ], Z_stoch)
        stoch_zeigen.append(stoch_results)
    obs_zeigen = jnp.concatenate(obs_zeigen, axis=0)
    h_zeigen = jnp.concatenate(h_zeigen, axis=0)
    stoch_zeigen = jnp.concatenate(stoch_zeigen, axis=0)
    mean_zeigen_obs = jnp.mean(obs_zeigen, axis=0)
    mean_zeigen_h = jnp.mean(h_zeigen, axis=0)
    mean_zeigen_stoch = jnp.mean(stoch_zeigen, axis=0)
    return mean_zeigen_obs, mean_zeigen_h, mean_zeigen_stoch


class Onestage_MCMC:
    def __init__(self, Y, X, Z_obs, Z_h, Z_stoch, estimand_h, estimand_stoch,
                 zeigen, h_zeigen, stoch_zeigen, rng_key, iter):
        self.Y = Y
        self.X = X
        self.Z_obs = Z_obs
        self.n = N
        self.Z_h = Z_h
        self.Z_stoch = Z_stoch
        self.estimand_h = estimand_h
        self.estimand_stoch = estimand_stoch
        self.zeigen = zeigen
        self.h_zeigen = h_zeigen
        self.stoch_zeigen = stoch_zeigen
        self.rng_key = rng_key
        self.iter = iter
        self.df = self.get_df()
        self.ell = jnp.array(C*jnp.max(jnp.abs(self.df[:,3]))).reshape(1,1)
        self.linear_post_samples = linear_model_samples_parallel(key=self.rng_key, Y=self.Y, df=self.df)
        self.hsgp_post_samples = HSGP_model_samples_parallel(key=self.rng_key, Y=self.Y, Xgp=self.df[:,3],
                                                             Xlin=self.df[:,0:3], ell=self.ell)
    def get_df(self):
        return jnp.transpose(jnp.array([[1]*self.n, self.Z_obs, self.X, self.zeigen]))

    def get_results(self):
        # dynamic (h) intervention
        linear_h_pred = linear_pred(self.Z_h, self.h_zeigen, self.linear_post_samples,
                                    self.X, self.rng_key)
        linear_h_stats = compute_error_stats(jnp.mean(linear_h_pred, axis=0),
                                                  self.estimand_h, idx=self.iter)
        hsgp_h_pred = hsgp_pred(self.Z_h, self.h_zeigen, self.hsgp_post_samples, self.X,
                                self.ell, self.rng_key)
        hsgp_h_stats = compute_error_stats(jnp.mean(hsgp_h_pred, axis=0),
                                                  self.estimand_h, idx=self.iter)
        # stochastic intervention
        linear_stoch_pred = linear_pred(self.Z_stoch, self.stoch_zeigen,
                                        self.linear_post_samples, self.X, self.rng_key)
        linear_stoch_stats = compute_error_stats(jnp.mean(linear_stoch_pred, axis=0),
                                             self.estimand_h, idx=self.iter)
        hsgp_stoch_pred = hsgp_pred(self.Z_stoch, self.stoch_zeigen,
                                    self.hsgp_post_samples, self.X,
                                    self.ell, self.rng_key)
        hsgp_stoch_stats = compute_error_stats(jnp.mean(hsgp_stoch_pred, axis=0),
                                           self.estimand_h, idx=self.iter)

        return jnp.vstack([linear_h_stats, hsgp_h_stats, linear_stoch_stats, hsgp_stoch_stats])
        # return pd.concat([linear_h_stats, hsgp_h_stats, linear_stoch_stats, hsgp_stoch_stats])


