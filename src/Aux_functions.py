# Load libraries
import time
import numpy as np
import pandas as pd
import os
from itertools import combinations
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
from jax.scipy.special import expit, logit
import numpyro.distributions as dist
import numpyro
from numpyro.contrib.funsor import config_enumerate
# import
from numpyro.infer import MCMC, NUTS, Predictive
from hsgp.approximation import hsgp_squared_exponential, eigenfunctions
from hsgp.spectral_densities import diag_spectral_density_squared_exponential
import pyro
import pyro.contrib.gp as gp
import torch
from tqdm import tqdm
from jax2torch import jax2torch


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# --- Set cores and seed ---
N_CORES = 4
# N_CORES = 20
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_CORES}"
# RANDOM_SEED = 892357143
# rng = np.random.default_rng(RANDOM_SEED)

# --- Set global variables and values ---
N = 300
TRIL_DIM = int(N*(N-1)/2)
M = 20
C = 3

# --- data generation for each iteration ---
class DataGeneration:
    def __init__(self, rng, theta, eta, sig_y, pz, lin, alphas, n=N):
        self.n = n
        self.theta = theta
        self.eta = eta
        self.sig_y = sig_y
        self.lin = lin
        self.alphas = alphas
        self.rng = rng
        self.X = self.generate_X()
        self.X2 = self.generate_X2()
        self.U_latent = self.generate_U_latent()
        self.Z = self.generate_Z(p=pz)
        self.X_diff = self.x_diff()
        self.X2_equal = self.x2_equal()
        self.U_diff_norm = self.latent_to_norm_of_diff()
        self.triu_dim = int(self.n*(self.n-1)/2)
        self.triu = self.generate_triu()
        self.adj_mat = self.generate_adj_matrix()
        self.eig_cen = eigen_centrality(self.adj_mat)
        self.zeigen = zeigen_value(self.Z, self.eig_cen, self.adj_mat)
        self.Y, self.epsi = self.gen_outcome(self.Z, self.zeigen, with_epsi=True)
        self.Z_h = self.dynamic_intervention()
        self.Z_stoch = self.stochastic_intervention()
        self.estimand_h = self.get_true_estimand(self.Z_h)
        self.estimand_stoch = self.get_true_estimand(self.Z_stoch)
        # self.exposures = self.generate_exposures()

    def generate_X(self, loc=0, scale=3):
        # return rng.normal(loc=loc, scale=scale, size=self.n)
        # return jnp.array(rng.normal(loc=loc, scale=scale, size=self.n))
        return jnp.array(self.rng.normal(loc=loc, scale=scale, size=self.n))

    def generate_X2(self,p=0.1):
        # return jnp.array(rng.binomial(n=1, p=p, size=self.n))
        return jnp.array(self.rng.binomial(n=1, p=p, size=self.n))

    def generate_U_latent(self, loc=0, scale=1, K=2):
        return jnp.array(self.rng.normal(loc=loc, scale=scale, size=(self.n, K)))

    def generate_Z(self, p):
        # return rng.binomial(n=1, p=p, size=self.n)
        # return jnp.array(rng.binomial(n=1, p=p, size=self.n))
        return jnp.array(self.rng.binomial(n=1, p=p, size=self.n))

    def x_diff(self):
        idx_pairs = list(combinations(range(self.n), 2))
        # x_d = np.array([abs(self.X[i] - self.X[j]) for i, j in idx_pairs])
        x_d = jnp.array([abs(self.X[i] - self.X[j]) for i, j in idx_pairs])
        return x_d

    def x2_equal(self):
        idx_pairs = list(combinations(range(self.n), 2))
        # x2_equal = np.array([1 if self.X2[i] == 1 and self.X2[j] == 1 else 0 for i, j in idx_pairs])
        # x2_equal = jnp.array([1 if (self.X2[i] == 1 or self.X2[j] == 1) else 0 for i, j in idx_pairs])
        x2_equal = jnp.array([1 if (self.X2[i] + self.X2[j] == 1) else 0 for i, j in idx_pairs])
        return x2_equal

    def latent_to_norm_of_diff(self):
        idx = jnp.triu_indices(n=self.n, k=1)
        U_diff = self.U_latent[idx[0]] - self.U_latent[idx[1]]
        return jnp.linalg.norm(U_diff, axis=1)

    def generate_triu(self):
        # probs = expit(self.theta[0] + self.theta[1]*self.X_diff  + self.theta[2]*self.X2_equal)
        probs = expit(self.theta[0] + self.theta[1]*self.X2_equal - self.U_diff_norm)
        # return jnp.array(rng.binomial(n=1, p=probs, size=self.triu_dim))
        # return rng.binomial(n=1, p=probs, size=self.triu_dim)
        return self.rng.binomial(n=1, p=probs, size=self.triu_dim)

    def generate_adj_matrix(self):
        mat = np.zeros((self.n, self.n))
        idx_upper_tri = np.triu_indices(n=self.n, k=1)
        mat[idx_upper_tri] = self.triu
        return mat + mat.T

    def f_zeigen(self, zeig, param):
        # Conditions
        cond1 = zeig < 1.2
        # cond2 = (zeig >= .6) & (zeig < 1)
        # Functions
        # f1 = 0
        # f2 = 10 * param * (zeig - .8)
        # f3 = 10 * param * (1 - .8)
        f2 = jnp.maximum(5 * param * (zeig - .6) ,0)
        f3 = jnp.maximum(5*param*(1.2-0.6) - 2.5*param*(zeig-1.2), 0)
        # Using jnp.where to implement piecewise function
        # result = jnp.where(cond1, f1, jnp.where(cond2, f2, f3))
        result = jnp.where(cond1, f2, f3)
        return result

    def gen_outcome(self, z, zeig, with_epsi):
        df_lin = jnp.transpose(np.array([[1]*self.n, z, self.X, z*self.X]))
        if self.lin:
            mean_y = jnp.dot(jnp.column_stack((df_lin, zeig)), self.eta)
            # mean_y = np.dot(np.column_stack((df_lin, zeig)), self.eta)
        else:
            # mean_lin = jnp.dot(df_lin, self.eta[0:4])
            # mean_nonlin = self.eta[3] / (1 + jnp.exp(-15 * (zeig - 0.4)))
            # mean_nonlin = self.eta[3]*(jnp.sin(20 * zeig) + jnp.log(zeig + 1))
            # mean_nonlin = 1.2*self.eta[4]*(jnp.sin(4 * (zeig - jnp.pi)) + jnp.log(zeig + 1))
            # mean_nonlin = jnp.exp(self.eta[4]*zeig)/40
            # mean_nonlin = -3*self.eta[4]*jnp.power(zeig-.8,2) + 3*self.eta[4]*zeig
            # mean_nonlin = 2*self.eta[4] / (1+jnp.exp(-5*zeig + 7.5))
            # mean_nonlin = self.f_zeigen(zeig, self.eta[4])
            # mean_nonlin = jnp.maximum(-2*self.eta[4]*jnp.power(zeig-1.2,2) + 2*self.eta[4]*zeig, 0)
            mean_nonlin = jnp.maximum(0, jnp.minimum(4*self.eta[4]*(zeig-.4), 4*self.eta[4]*0.3))
            mean_nonlin += 4 * z * zeig
            # mean_nonlin += self.eta[4] * z * zeig
            mean_lin = jnp.dot(df_lin, self.eta[0:4])
            mean_y = mean_lin + mean_nonlin + 0.25*jnp.power(z*self.X,2)
        if with_epsi:
            # epsi = jnp.array(rng.normal(loc=0, scale=self.sig_y, size=self.n))
            epsi = jnp.array(self.rng.normal(loc=0, scale=self.sig_y, size=self.n))
            Y = jnp.array(mean_y + epsi)
            return Y, epsi
        else:
            return mean_y

    def dynamic_intervention(self, thresholds=(1.5, 2)):
        Z_h1 = jnp.where((self.X > thresholds[0]) | (self.X < -thresholds[0]), 1, 0)
        Z_h2 = jnp.where((self.X > thresholds[1]) | (self.X < -thresholds[1]), 1, 0)
        # Z_h1 = jnp.where(((self.X >= -1) & (self.X <= 1)) | (self.X2 == 1), 1, 0)
        # Z_h2 = jnp.where(self.X2 == 1, 1, 0)
        return jnp.array([Z_h1, Z_h2])
        # return jnp.where((self.X > threshold) | (self.X < -threshold), 1, 0)

    def stochastic_intervention(self, n_approx=100):
    # def stochastic_intervention(self, n_approx=2000):
        z_stoch1 = self.rng.binomial(n=1, p=self.alphas[0], size=(n_approx, self.n))
        # z_stoch1 = rng.binomial(n=1, p=self.alphas[0], size=(n_approx, self.n))
        # z_stoch2 = rng.binomial(n=1, p=self.alphas[1], size=(n_approx, self.n))
        z_stoch2 = self.rng.binomial(n=1, p=self.alphas[1], size=(n_approx, self.n))
        return jnp.array([z_stoch1, z_stoch2])
        # return rng.binomial(n=1, p=self.alpha, size=(n_approx, self.n))

    def get_true_estimand(self, z_new):
        if z_new.ndim == 3:
            zeigen_new1 = zeigen_value(z_new[0,:,:], self.eig_cen, self.adj_mat)
            # zeigen_new2 = zeigen_value(z_new[1,:,:], self.eig_cen, self.adj_mat)
            n_stoch = z_new.shape[1]
            results = np.zeros((n_stoch, N))
            for i in range(n_stoch):
                # df = np.transpose(np.array([[1] * self.n, z_new[i,], self.X]))
                y1 = self.gen_outcome(z_new[0,i,], zeigen_new1[i,], with_epsi=False)
                # y2 = self.gen_outcome(z_new[1,i,], zeigen_new2[i,], with_epsi=False)
                # results[i,] = jnp.mean(y1 - epsi2) - jnp.mean(y2 - epsi2)
                # results[i,] = jnp.mean(y1 - y2)
                # results[i,] = jnp.mean(y1)
                results[i,] = y1
                # results[i,] = y1 - y2
            return jnp.mean(results, axis=0).squeeze()
        else:
            # assert Z_stoch.ndim == 2
            zeigen_new1 = zeigen_value(z_new[0,:], self.eig_cen, self.adj_mat)
            # zeigen_new2 = zeigen_value(z_new[1,:], self.eig_cen, self.adj_mat)
            y1 = self.gen_outcome(z_new[0,], zeigen_new1, with_epsi=False)
            # y2 = self.gen_outcome(z_new[1,], zeigen_new2, with_epsi=False)
            # return jnp.mean(y1 - epsi1) - jnp.mean(y2 - epsi2)
            # return jnp.mean(y1 - y2)
            # return jnp.mean(y1)
            return y1
            # return y1 - y2

    def get_data(self):
        return {"Z" : self.Z, "X" : self.X,
                "X2" : self.X2, "Y" : self.Y,
                "triu" : self.triu, "Zeigen" : self.zeigen,
                "eig_cen" : self.eig_cen, "adj_mat" : self.adj_mat,
                "X_diff" : self.X_diff, "X2_equal" : self.X2_equal,
                "Z_h" : self.Z_h, "Z_stoch" : self.Z_stoch,
                "estimand_h" : self.estimand_h,
                "estimand_stoch" : self.estimand_stoch}


# --- General aux functions ---
def create_noisy_network(rng, triu_vals, gamma, x_diff):
    obs_mat = np.zeros((N, N))  # create nXn matrix of zeros
    triu_idx = np.triu_indices(n=N, k=1)
    # logit_nois = triu_vals*logit(gamma[0]) + (1 - triu_vals)*(gamma[1] + gamma[2]*x2_equal)
    # logit_nois = triu_vals*logit(gamma[0]) + (1 - triu_vals)*(gamma[1]*x_diff)
    logit_nois = triu_vals*gamma[0] + (1-triu_vals)*(gamma[1] + gamma[2]*x_diff)
    # edges_noisy = rng.binomial(n=1, p=expit(logit_nois), size=TRIL_DIM)
    edges_noisy = rng.binomial(n=1, p=expit(logit_nois), size=TRIL_DIM)
    obs_mat[triu_idx] = edges_noisy
    obs_mat = obs_mat + obs_mat.T
    # triu_obs = obs_mat[triu_idx]
    return {"obs_mat" : obs_mat,
            "triu_obs" : edges_noisy}

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


def compute_hdi(samples, cred_mass):
    sorted_samples = jnp.sort(samples)
    interval_size = int(jnp.ceil(cred_mass * len(sorted_samples)))

    min_width = jnp.inf
    hdi_lower = 0
    hdi_upper = 0

    for i in range(len(sorted_samples) - interval_size):
        current_width = sorted_samples[i + interval_size] - sorted_samples[i]
        if current_width < min_width:
            min_width = current_width
            hdi_lower = sorted_samples[i]
            hdi_upper = sorted_samples[i + interval_size]

    return hdi_lower, hdi_upper


# def compute_error_stats(esti_post_draws, true_estimand, method="TEST", estimand = "h", idx=None):
def compute_error_stats(esti_post_draws, true_estimand, idx=None):
    # esti_post_draws has shape (M, N)
    # true_estimand has shape (N,)
    mean_estimand = jnp.mean(true_estimand)  # scalar
    mean_units = jnp.mean(esti_post_draws, axis=1)  # shape (M,)
    mean_samples = jnp.mean(esti_post_draws, axis=0)  # shape (N,)
    mean_all = jnp.round(jnp.mean(esti_post_draws), 3)  # scalar
    # mean = jnp.round(jnp.mean(esti_post_draws),3)
    medi = jnp.round(jnp.median(mean_units),3)
    std = jnp.round(jnp.std(mean_units),3)
    # RMSE_all = jnp.round(jnp.sqrt(jnp.mean(jnp.power(esti_post_draws - true_estimand, 2))),3)
    RMSE = jnp.round(jnp.sqrt(jnp.mean(jnp.power(mean_samples - true_estimand, 2))),3)
    # MAE_all = jnp.round(jnp.mean(jnp.abs(esti_post_draws - true_estimand)), 3)
    MAE = jnp.round(jnp.mean(jnp.abs(mean_samples - true_estimand)),3)
    MAPE = jnp.round(jnp.mean(jnp.abs((mean_samples - true_estimand) / true_estimand)), 3)
    q025 = jnp.quantile(mean_units, 0.025)
    q025_ind = jnp.quantile(esti_post_draws, 0.025, axis=0)
    q975 = jnp.quantile(mean_units, 0.975)
    q975_ind = jnp.quantile(esti_post_draws, 0.975, axis=0)
    # hdi_lower, hdi_upper = compute_hdi(esti_post_draws, 0.95)
    cover = (q025 <= mean_estimand) & (mean_estimand <= q975)
    mean_cover = (q025_ind <= true_estimand) & (true_estimand <= q975_ind)
    # cover = q025 <= true_estimand <= q975
    return jnp.array([idx, mean_all, medi, jnp.round(mean_estimand,3),
                      jnp.round(mean_all - mean_estimand,3), std,
                      RMSE, MAE, MAPE,
                      jnp.round(q025,3), jnp.round(q975,3),
                      cover, jnp.mean(mean_cover)])
                      # , jnp.round(hdi_lower,3), jnp.round(hdi_upper,3)])
 # return pd.DataFrame([{"idx" : idx, "method" : method, "estimand" : estimand,
 #            "mean" : mean, "median" : medi, "true" : jnp.round(true_estimand,3),
 #            "bias" : jnp.round(mean - true_estimand,3), "std" : std, "RMSE" : RMSE,
 #            "q025" : jnp.round(q025,3), "q975" : jnp.round(q975,3), "covering" : cover}])


# --- NumPyro and pyro models ---

@pyro.infer.config_enumerate
def pyro_noisy_networks_model(x, x2, triu_v, N, K=2, eps=1e-3):
    """
    Network model for one noisy observed network. True network is geenrated from LSM.
    :param x: pairwise x-differences
    :param x2: pariwise x2-equality
    :param triu_v: observed triu values (upper triangular)
    :param N: number of units
    :param K: latent variables dimension
    """
    # log_sigma_sq = pyro.sample("log_sigma_sq", dist.Normal(0, 5))
    # sigma_sq = torch.exp(log_sigma_sq)
    with pyro.plate("Latent_dim", N):
        nu = pyro.sample("nu",
                         pyro.distributions.MultivariateNormal(torch.zeros(K) + eps, torch.eye(K)))
        # nu_standard = pyro.sample("nu_standard", dist.MultivariateNormal(torch.zeros(K), torch.eye(K)))
    # nu = pyro.deterministic("nu", nu_standard * torch.sqrt(sigma_sq))

    idx = torch.triu_indices(N, N, offset=1)
    nu_diff = nu[idx[0]] - nu[idx[1]]
    nu_diff_norm_val = torch.norm(nu_diff, dim=1)

    with pyro.plate("theta_dim", 2):
        theta = pyro.sample("theta",
                            pyro.distributions.Normal(0, 5))

    mu_net = theta[0] + x2 * theta[1] - nu_diff_norm_val
    mu_net = torch.clamp(mu_net, min=-30, max=30)

    with pyro.plate("gamma_i", 3):
        gamma = pyro.sample("gamma",
                            pyro.distributions.Normal(0, 5))

    with pyro.plate("A* and A", x.shape[0]):
        triu_star = pyro.sample("triu_star",
                                pyro.distributions.Bernoulli(logits=mu_net),
                                infer={"enumerate": "parallel"})

        logit_misspec = torch.where(triu_star == 1.0,
                                    gamma[0],
                                    gamma[1] + gamma[2] * x)

        pyro.sample("obs_triu",
                    pyro.distributions.Bernoulli(logits=logit_misspec),
                    obs=triu_v)


@config_enumerate
def network_model(X_d, X2_eq, triu_v):
    # Network model
    with numpyro.plate("theta_i", 2):
        theta = numpyro.sample("theta", dist.Normal(0, 5))
    mu_net = theta[0] + theta[1]*X2_eq

    with numpyro.plate("gamma_i", 2):
        gamma = numpyro.sample("gamma", dist.Normal(0, 5))

    with numpyro.plate("A* and A", triu_v.shape[0]):
        triu_star = numpyro.sample("triu_star", dist.Bernoulli(logits=mu_net),
                                   infer={"enumerate": "parallel"})
        # logit_misspec = (triu_star*gamma[0]) + ((1 - triu_star)*(gamma[1] + gamma[2]*X2_eq))
        logit_misspec = triu_star*gamma[0] + (1 - triu_star)*(gamma[1]*X_d)
        numpyro.sample("obs_triu", dist.Bernoulli(logits=logit_misspec), obs=triu_v)


def outcome_model(df, Y=None):
    # --- priors ---
    with numpyro.plate("Lin coef.", df.shape[1]):
        eta = numpyro.sample("eta", dist.Normal(0, 5))
    sig = numpyro.sample("sig", dist.HalfNormal(scale=2))
    mu_y = jnp.dot(df, eta)
    # --- likelihood --
    with numpyro.plate("obs", df.shape[0]):
        numpyro.sample("Y", dist.Normal(loc=mu_y, scale=sig), obs=Y)


def HSGP_model(df1, df2, ell1, ell2, m, Y=None):
    # --- Priors ---
    amplitude1 = numpyro.sample("amplitude1", dist.HalfNormal(2))
    length1 = numpyro.sample("lengthscale1", dist.HalfNormal(5))
    amplitude2 = numpyro.sample("amplitude2", dist.HalfNormal(2))
    length2 = numpyro.sample("lengthscale2", dist.HalfNormal(5))
    noise = numpyro.sample("noise", dist.HalfNormal(2))
    # --- GP ---
    f1 = numpyro.deterministic(
        "f1_star",
        hsgp_squared_exponential(
            df1, alpha=amplitude1, length=length1, ell=ell1, m=m, i="1"
        ),
    )
    f2 = numpyro.deterministic(
        "f2_star",
        hsgp_squared_exponential(
            df2, alpha=amplitude2, length=length2, ell=ell2, m=m, i="2"
        ),
    )
    # --- Likelihood ---
    # intercept = numpyro.sample("intercept",dist.Normal(0,5))
    # f = numpyro.deterministic("f_star", intercept + f1 + f2)
    f = numpyro.deterministic("f_star", f1 + f2)
    numpyro.sample("Y", dist.Normal(f, noise), obs=Y)


# def HSGP_model(Xlin, Xgp, ell, m, Y=None, non_centered=True):
#     # --- Priors ---
#     magn = numpyro.sample("magn", dist.HalfNormal(2))
#     length = numpyro.sample("length", dist.HalfNormal(5))
#     sig = numpyro.sample("sig", dist.HalfNormal(2))
#     # --- GP ---
#     f = hsgp_squared_exponential(
#         x=Xgp, alpha=magn, length=length, ell=ell, m=m, non_centered=non_centered)
#     # --- Linear part ---
#     with numpyro.plate("Lin coef.", Xlin.shape[1]):
#         eta = numpyro.sample("eta", dist.Normal(0, 5))
#     # --- Combine linear with GP ---
#     mu = jnp.dot(Xlin, eta) + f
#     # --- Likelihood ---
#     with numpyro.plate("obs", Xlin.shape[0]):
#         numpyro.sample("Y", dist.Normal(loc=mu, scale=sig), obs=Y)


# --- MCMC aux functions ---
# @jit
def linear_model_samples_parallel(key, Y, df):
    kernel_outcome = NUTS(outcome_model)
    # lin_mcmc = MCMC(kernel_outcome, num_warmup=1000, num_samples=2000,num_chains=4, progress_bar=False)
    lin_mcmc = MCMC(kernel_outcome, num_warmup=2000, num_samples=2500,num_chains=4, progress_bar=False)
    lin_mcmc.run(key, df=df, Y=Y)
    return lin_mcmc.get_samples()

@jit
def linear_model_samples_vectorized(key, Y, df):
    kernel_outcome = NUTS(outcome_model)
    # lin_mcmc = MCMC(kernel_outcome, num_warmup=600, num_samples=150, num_chains=1,
    lin_mcmc = MCMC(kernel_outcome, num_warmup=1000, num_samples=100, num_chains=1,
                    progress_bar=False, chain_method="vectorized")
    lin_mcmc.run(key, df=df, Y=Y)
    return lin_mcmc.get_samples()


@jit
def outcome_jit_pred(post_samples, df_arr, key):
    pred_func = Predictive(outcome_model, post_samples)
    return pred_func(key, df_arr)

# @jit
def HSGP_model_samples_parallel(key, Y, df1, df2, ell1, ell2):
    kernel_hsgp = NUTS(HSGP_model, target_accept_prob=0.9)
    # hsgp_mcmc = MCMC(kernel_hsgp, num_warmup=1000, num_samples=2000,num_chains=4, progress_bar=False)
    hsgp_mcmc = MCMC(kernel_hsgp, num_warmup=2000, num_samples=2500,num_chains=4, progress_bar=False)
    hsgp_mcmc.run(key, df1=df1, df2=df2, ell1=ell1, ell2=ell2 ,m=M, Y=Y)
    return hsgp_mcmc.get_samples()

@jit
def HSGP_model_samples_vectorized(key, Y, df1, df2, ell1, ell2):
    kernel_hsgp = NUTS(HSGP_model, target_accept_prob=0.9)
    # hsgp_mcmc = MCMC(kernel_hsgp, num_warmup=600, num_samples=150,num_chains=1,
    hsgp_mcmc = MCMC(kernel_hsgp, num_warmup=1000, num_samples=100,num_chains=1,
                     progress_bar=False, chain_method="vectorized")
    hsgp_mcmc.run(key, df1=df1, df2=df2, ell1=ell1, ell2=ell2 ,m=M, Y=Y)
    return hsgp_mcmc.get_samples()

@jit
def HSGP_jit_pred(post_samples, df1, df2, ell1, ell2, key):
    pred_func = Predictive(HSGP_model, post_samples)
    return pred_func(key, df1=df1, df2=df2, ell1=ell1, ell2=ell2 ,m=M)

@jit
def manual_linear_pred(samples, df):
    return jnp.dot(samples["eta"], jnp.transpose(df))


def compute_f_star(df, ell, amplt, length, beta):
    dim = df.shape[-1] if df.ndim > 1 else 1
    phi_new = eigenfunctions(x=df, ell=ell, m=M)

    def compute_single(alpha, length, beta):
        spd_post = jnp.sqrt(
            diag_spectral_density_squared_exponential(
                alpha=alpha, length=length, ell=ell, m=M, dim=dim
            )
        )
        return phi_new @ (spd_post * beta)
    # Vectorize the computation across the first dimension of amplt and beta
    compute_single_vectorized = vmap(compute_single, in_axes=(0, 0, 0))
    f_res = compute_single_vectorized(amplt, length, beta)
    return f_res

def manual_gp_f_star_pred(df1, df2, ell1, ell2, post_samples):
    f_1 = compute_f_star(df1, ell1,
                         post_samples["amplitude1"],
                         post_samples["lengthscale1"],
                         post_samples["beta1"])
    f_2 = compute_f_star(df2, ell2,
                         post_samples["amplitude2"],
                         post_samples["lengthscale2"],
                         post_samples["beta2"])
    return f_1 + f_2

@jit
def linear_model_outcome_pred(z, zeigen, post_samples, x, x2, key):
    # df = jnp.transpose(jnp.array([[1] * N, z, x, x2, zeigen]))
    df = jnp.transpose(jnp.array([[1] * N, z, x, z*x, zeigen]))
    # df = jnp.transpose(jnp.array([[1] * N, z, zeigen]))
    # pred = outcome_jit_pred(post_samples, df, key)
    # pred = manual_linear_pred(post_samples, df)
    # return jnp.mean(pred["Y"], axis=1)
    return manual_linear_pred(post_samples, df)

linear_model_pred = vmap(linear_model_outcome_pred, in_axes=(0, 0, None, None, None, None))

def linear_pred(z, zeigen, post_samples, x, x2, key):
    if z.ndim == 2:
        # return linear_model_pred(z, zeigen, post_samples, x, x2, key)
        return jnp.mean(linear_model_pred(z, zeigen, post_samples, x, x2, key), axis=0)
    if z.ndim == 1:
        n_z = z.shape[0]
        # return linear_model_pred(z.reshape((1, n_z)), zeigen.reshape((1, n_z)), post_samples, x, x2, key)
        return linear_model_pred(z.reshape((1, n_z)), zeigen.reshape((1, n_z)), post_samples, x, x2, key).squeeze()

@jit
def hsgp_model_outcome_pred(z, zeigen, post_samples, x, x2, ell1, ell2):
    # ell_ = jnp.array(c*jnp.max(jnp.abs(zeigen))).reshape(1,1)
    # df = jnp.transpose(jnp.array([[1] * x.shape[0], z, x, x2, zeigen]))
    df1 = jnp.transpose(jnp.array([z, x]))
    df2 = jnp.transpose(jnp.array([z, zeigen]))
    # df = jnp.transpose(jnp.array([[1] * N, z, zeigen]))
    # pred = HSGP_jit_pred(post_samples, Xgp=df[:, 4:], Xlin=df[:, 0:4], ell=ell, key=key)
    # pred = HSGP_jit_pred(post_samples, Xgp=df[:, 3:], Xlin=df[:, 0:3], ell=ell, key=key)
    # pred = HSGP_jit_pred(post_samples, Xgp=df[:, 2:], Xlin=df[:, 0:2], ell=ell, key=key)
    # return jnp.mean(pred["Y"], axis=1)
    return manual_gp_f_star_pred(df1, df2,ell1, ell2, post_samples)

hsgp_model_pred = vmap(hsgp_model_outcome_pred, in_axes=(0, 0, None, None, None, None, None))

def hsgp_pred(z, zeigen, post_samples, x, x2, ell1, ell2):
    if z.ndim == 2:
        # return hsgp_model_pred(z, zeigen, post_samples, x, x2, ell, key)
        return jnp.mean(hsgp_model_pred(z, zeigen, post_samples, x, x2, ell1, ell2), axis=0)
    if z.ndim == 1:
        n_z = z.shape[0]
        # return hsgp_model_pred(z.reshape((1, n_z)), zeigen.reshape((1, n_z)), post_samples, x, x2, ell, key)
        return hsgp_model_pred(z.reshape((1, n_z)), zeigen.reshape((1, n_z)), post_samples, x, x2, ell1, ell2).squeeze()


@jit
def Astar_pred(i, post_samples, Xd, X2_eq, triu):
    pred_func = Predictive(model=network_model, posterior_samples=post_samples, infer_discrete=True, num_samples=1)
    sampled_net = pred_func(random.PRNGKey(i), X_d=Xd, X2_eq=X2_eq, TriU=triu)["triu_star"]
    return jnp.squeeze(sampled_net, axis=0)

vectorized_astar_pred = vmap(Astar_pred, in_axes=(0, None, None, None, None))
parallel_astar_pred = pmap(Astar_pred, in_axes=(0, None, None, None, None))


def get_many_post_astars(K, post_pred_mean, x_diff, x2_eq, triu_v):
    i_range = jnp.arange(K)
    return vectorized_astar_pred(i_range, post_pred_mean, x_diff, x2_eq, triu_v)


class Network_SVI:
    def __init__(self, data, rng_key, n_iter=2000, n_samples=1000, network_model=pyro_noisy_networks_model):
        self.X_diff = torch.tensor(np.array(data["X_diff"]), dtype=torch.float32)
        self.X2_eq = torch.tensor(np.array(data["X2_equal"]), dtype=torch.float32)
        self.triu = torch.tensor(np.array(data["triu"]), dtype=torch.float32)
        self.n = N
        # self.adj_mat = data["adj_mat"]
        self.rng_key = rng_key
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.network_model = network_model
        self.guide = self.get_guide()

    def get_guide(self):
        return pyro.infer.autoguide.AutoLowRankMultivariateNormal(pyro.poutine.block(pyro_noisy_networks_model,
                                                              hide=["triu_star"]),
                                      init_loc_fn = pyro.infer.autoguide.init_to_median())

    def train_model(self):
        pyro.clear_param_store()
        loss_func = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)
        optimzer = pyro.optim.ClippedAdam({"lr": 0.001})
        svi = pyro.infer.SVI(self.network_model, self.guide, optimzer, loss=loss_func)
        # losses_full = []
        for _ in tqdm(range(self.n_iter), desc="Training network model"):
            svi.step(self.X_diff, self.X2_eq, self.triu, self.n)
            # loss = svi.step(self.X_diff, self.X2_eq, self.triu, self.n)
            # losses_full.append(loss)

    def network_samples(self):
        triu_star_samples = []
        for _ in tqdm(range(self.n_samples), desc="Sampling A*"):
            # Get a trace from the guide
            guide_trace = pyro.poutine.trace(self.guide).get_trace(self.X_diff, self.X2_eq, self.triu, self.n)
            # Run infer_discrete
            inferred_model = pyro.infer.infer_discrete(pyro.poutine.replay(self.network_model, guide_trace),
                                            first_available_dim=-2)
            # Get a trace from the inferred model
            model_trace = pyro.poutine.trace(inferred_model).get_trace(self.X_diff, self.X2_eq, self.triu, self.n)
            # Extract triu_star from the trace
            triu_star_samples.append(model_trace.nodes['triu_star']['value'])
        # Convert to tensor
        return jnp.stack(jnp.array(triu_star_samples))
        # triu_star_samples = torch.stack(triu_star_samples)
        # return jnp.array(triu_star_samples)

class Network_MCMC:
    # def __init__(self, data, rng_key, n_warmup=1000, n_samples=1500, n_chains=4):
    def __init__(self, data, rng_key, n_warmup=2000, n_samples=4000, n_chains=4):
    # def __init__(self, data, n, rng_key, n_warmup=1000, n_samples=2000, n_chains=6):
        self.X_diff = data["X_diff"]
        self.X2_eq = data["X2_equal"]
        self.triu = data["triu"]
        self.adj_mat = data["adj_mat"]
        self.rng_key = rng_key
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.network_m = self.network()
        self.post_samples = None

    def network(self):
        kernel = NUTS(network_model,
                      target_accept_prob=0.9,
                      init_strategy=numpyro.infer.init_to_median(num_samples=30))
        return MCMC(kernel, num_warmup=self.n_warmup, num_samples=self.n_samples,
                    num_chains=self.n_chains, progress_bar=False)

    def get_posterior_samples(self):
        self.network_m.run(self.rng_key, X_d=self.X_diff, X2_eq=self.X2_eq, triu_v=self.triu)
        self.network_m.print_summary()
        self.post_samples = self.network_m.get_samples()
        return self.post_samples

    # def mean_posterior(self):
    #     return {"theta": jnp.expand_dims(jnp.mean(self.post_samples["theta"], axis=0), -2),
    #                           "gamma0": jnp.expand_dims(np.mean(self.post_samples["gamma0"]), -1),
    #                           "gamma1": jnp.expand_dims(jnp.mean(self.post_samples["gamma1"]), -1)}

    def predictive_samples(self):
        posterior_predictive = Predictive(model=network_model, posterior_samples=self.post_samples,
                                          infer_discrete=True)
        return posterior_predictive(self.rng_key, X_d=self.X_diff, X2_eq=self.X2_eq, triu_v=self.triu)["triu_star"]


class Outcome_MCMC:
    def __init__(self, data, type, rng_key, iter):
        self.X = data["X"]
        self.X2 = data["X2"]
        self.Z = data["Z"]
        self.Y = data["Y"]
        self.adj_mat = data["adj_mat"]
        self.eig_cen = eigen_centrality(self.adj_mat)
        self.zeigen = zeigen_value(self.Z, self.eig_cen, self.adj_mat)
        self.n = N
        # self.ell = jnp.array(C*jnp.max(jnp.abs(self.zeigen))).reshape(1,1)
        # self.ell1 = [C, C*jnp.max(jnp.abs(self.X))]
        # self.ell2 = [C, C*jnp.max(jnp.abs(self.zeigen))]
        self.df_lin = self.get_df()
        # self.df1 = jnp.transpose(jnp.array([self.Z, self.X]))
        # self.df2 = jnp.transpose(jnp.array([self.Z, self.zeigen]))
        self.Z_h = data["Z_h"]
        self.Z_stoch = data["Z_stoch"]
        self.estimand_h = data["estimand_h"]
        self.estimand_stoch = data["estimand_stoch"]
        self.type = type
        self.rng_key = rng_key
        self.iter = iter
        self.linear_post_samples = linear_model_samples_parallel(key=self.rng_key, Y=self.Y, df=self.df_lin)
        self.gp_reg = Outcome_GP(self.X, self.Y, self.Z, self.zeigen)
        self.gp_reg.train_model()
        # self.hsgp_post_samples = HSGP_model_samples_parallel(key=self.rng_key, Y=self.Y,
        #                                                      df1=self.df1, df2=self.df2,
        #                                                         ell1=self.ell1, ell2=self.ell2)
        self.print_zeig_error(data)

    def print_zeig_error(self, data):
        if self.type == "observed":
            print("Obs. mape zeigen: ",jnp.mean(jnp.abs(self.zeigen - data["Zeigen"])/jnp.abs(data["Zeigen"])))

    def get_df(self):
        # return jnp.transpose(jnp.array([[1]*self.n, self.Z, self.X, self.X2, self.zeigen]))
        # return jnp.transpose(jnp.array([[1]*self.n, self.Z, self.X, self.zeigen]))
        return jnp.transpose(jnp.array([[1]*self.n, self.Z, self.X, self.Z*self.X, self.zeigen]))
        # return jnp.transpose(jnp.array([[1]*self.n, self.Z, self.zeigen]))

    def get_results(self):
        # dynamic (h) intervention
        h1_zeigen = zeigen_value(self.Z_h[0,:], self.eig_cen, self.adj_mat)
        # h2_zeigen = zeigen_value(self.Z_h[1,:], self.eig_cen, self.adj_mat)

        linear_h1_pred = linear_pred(self.Z_h[0,:], h1_zeigen, self.linear_post_samples,
                                    self.X, self.X2, self.rng_key)
        # linear_h2_pred = linear_pred(self.Z_h[1,:], h2_zeigen, self.linear_post_samples,
        #                               self.X, self.X2, self.rng_key)
        # linear_h = jnp.mean(linear_h1_pred, axis=0) - jnp.mean(linear_h2_pred, axis=0)
        # linear_h = linear_h1_pred - linear_h2_pred
        linear_h = linear_h1_pred
        # linear_h = jnp.mean(linear_h1_pred - linear_h2_pred, axis=0)
        # linear_h = jnp.mean(linear_h1_pred, axis=0)
        linear_h_stats = compute_error_stats(linear_h, self.estimand_h, idx=self.iter)

        # hsgp_h1_pred = hsgp_pred(self.Z_h[0,:], h1_zeigen, self.hsgp_post_samples,
        #                         self.X, self.X2, self.ell1, self.ell2)
        # hsgp_h1_pred = hsgp_pred(self.Z_h[0,:], h1_zeigen, self.hsgp_post_samples,
        #                         self.X, self.X2,self.ell, self.rng_key)
        # hsgp_h2_pred = hsgp_pred(self.Z_h[1,:], h2_zeigen, self.hsgp_post_samples,
        #                           self.X, self.X2, self.ell, self.rng_key)
        # hsgp_h = jnp.mean(hsgp_h1_pred, axis=0) - jnp.mean(hsgp_h2_pred, axis=0)
        # hsgp_h = hsgp_h1_pred - hsgp_h2_pred
        # hsgp_h = hsgp_h1_pred
        # hsgp_h = hsgp_h1_pred
        # hsgp_h = jnp.mean(hsgp_h1_pred - hsgp_h2_pred, axis=0)
        # hsgp_h = jnp.mean(hsgp_h1_pred, axis=0)
        # hsgp_h_stats = compute_error_stats(hsgp_h, self.estimand_h, idx=self.iter)
        gp_h_pred = self.gp_reg.predict(self.Z_h[0,:], h1_zeigen)
        gp_h_stats = compute_error_stats(gp_h_pred, self.estimand_h, idx=self.iter)

        # stochastic intervention
        stoch_zeigen1 = zeigen_value(self.Z_stoch[0,:,:], self.eig_cen, self.adj_mat)
        # stoch_zeigen2 = zeigen_value(self.Z_stoch[1,:,:], self.eig_cen, self.adj_mat)

        linear_stoch_pred1 = linear_pred(self.Z_stoch[0,:,:], stoch_zeigen1,
                                        self.linear_post_samples, self.X, self.X2, self.rng_key)
        # linear_stoch_pred2 = linear_pred(self.Z_stoch[1,:,:], stoch_zeigen2,
        #                                  self.linear_post_samples, self.X, self.X2, self.rng_key)
        # linear_stoch_pred = jnp.mean(linear_stoch_pred1, axis=0) - jnp.mean(linear_stoch_pred2, axis=0)
        # linear_stoch_pred = linear_stoch_pred1 - linear_stoch_pred2
        linear_stoch_pred = linear_stoch_pred1
        # linear_stoch_pred = jnp.mean(linear_stoch_pred1 - linear_stoch_pred2, axis=0)
        # linear_stoch_pred = jnp.mean(linear_stoch_pred1, axis=0)
        linear_stoch_stats = compute_error_stats(linear_stoch_pred, self.estimand_stoch, idx=self.iter)

        # hsgp_stoch_pred1 = hsgp_pred(self.Z_stoch[0,:,:], stoch_zeigen1, self.hsgp_post_samples,
        #                             self.X, self.X2, self.ell1, self.ell2)
        # hsgp_stoch_pred1 = hsgp_pred(self.Z_stoch[0,:,:], stoch_zeigen1, self.hsgp_post_samples,
        #                             self.X, self.X2, self.ell, self.rng_key)
        # hsgp_stoch_pred2 = hsgp_pred(self.Z_stoch[1,:,:], stoch_zeigen2, self.hsgp_post_samples,
                                     # self.X, self.X2, self.ell, self.rng_key)
        # hsgp_stoch_pred = jnp.mean(hsgp_stoch_pred1, axis=0) - jnp.mean(hsgp_stoch_pred2, axis=0)
        # hsgp_stoch_pred = hsgp_stoch_pred1 - hsgp_stoch_pred2
        # hsgp_stoch_pred = hsgp_stoch_pred1
        # hsgp_stoch_pred = jnp.mean(hsgp_stoch_pred1 - hsgp_stoch_pred2, axis=0)
        # hsgp_stoch_pred = jnp.mean(hsgp_stoch_pred1, axis=0)
        # hsgp_stoch_stats = compute_error_stats(hsgp_stoch_pred,
        #                                    self.estimand_stoch, idx=self.iter)
        gp_stoch_pred = self.gp_reg.predict(self.Z_stoch[0,:,:], stoch_zeigen1)
        gp_stoch_stats = compute_error_stats(gp_stoch_pred,
                                           self.estimand_stoch, idx=self.iter)


        return jnp.vstack([linear_h_stats, gp_h_stats, linear_stoch_stats, gp_stoch_stats])
        # return jnp.vstack([linear_h_stats, hsgp_h_stats, linear_stoch_stats, hsgp_stoch_stats])
        # return pd.concat([linear_h_stats, hsgp_h_stats, linear_stoch_stats, hsgp_stoch_stats])

class Outcome_GP:
    def __init__(self, X, Y, Z, Zeigen, n_iter = 5000, n_samples = 10000):
        # self.X = torch.from_numpy(np.array(X))
        self.X = jax2torch(lambda x: x)(X)
        # self.X2 = torch.tensor(X2)
        # self.Y = torch.from_numpy(np.array(Y))
        self.Y = jax2torch(lambda x: x)(Y)
        self.Z = jax2torch(lambda x: x)(Z)
        # self.Z = torch.from_numpy(np.array(Z))
        # self.adj_mat = data["adj_mat"]
        # self.eig_cen = eigen_centrality(self.adj_mat)
        # self.zeigen = torch.from_numpy(np.array(zeigen_value(self.Z, self.eig_cen, self.adj_mat)))
        self.zeigen = jax2torch(lambda x: x)(Zeigen)
        # self.zeigen = torch.from_numpy(np.array(Zeigen))
        self.n = N
        self.df = self.get_df()
        self.gpr = self.gpr_model()
        # self.rng_key = rng_key
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.post_samples = None

    def get_df(self):
        return torch.stack([self.Z, self.zeigen, self.X], dim=1)
        # return torch.stack([torch.tensor(self.Z), self.zeigen, self.X], dim=1)

    def gpr_model(self):
        kernel = gp.kernels.RBF(input_dim=3, variance=torch.tensor(1.0), lengthscale=torch.tensor(1.0))
        return gp.models.GPRegression(self.df, self.Y, kernel, noise=torch.tensor(1.0))

    def train_model(self):
        pyro.clear_param_store()
        # Define priors on the hyperparameters
        self.gpr.kernel.lengthscale = pyro.nn.PyroSample(pyro.distributions.LogNormal(0.0, 3.0))
        self.gpr.kernel.variance = pyro.nn.PyroSample(pyro.distributions.LogNormal(0.0, 2.0))
        self.gpr.noise = pyro.nn.PyroSample(pyro.distributions.LogNormal(0.0, 1.0))
        # Set up the optimizer
        optimizer = torch.optim.Adam(self.gpr.parameters(), lr=0.001)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        # Training loop
        for i in tqdm(range(self.n_iter), desc="GP training"):
            optimizer.zero_grad()
            loss = loss_fn(self.gpr.model, self.gpr.guide)
            # loss = compute_loss(gpr.model, gpr.guide)
            loss.backward()
            optimizer.step()

    def predict_one_df(self, df):
        with torch.no_grad():
            mean, cov = self.gpr(df, full_cov=True, noiseless=False)
            # cov = cov + torch.eye(cov.shape[0]) * self.jitter
            # Add small jitter to ensure positive definiteness
            jitter = torch.eye(cov.shape[0]) * 1e-6
            L = torch.linalg.cholesky(cov + jitter)

            # Generate samples from standard normal distribution
            eps = torch.randn(cov.shape[0], self.n_samples)

            # Transform to samples from multivariate normal
            samples = mean.unsqueeze(1) + L @ eps

        return jnp.array(samples.T)
            # samples = pyro.distributions.MultivariateNormal(mean, cov).sample(sample_shape=(self.n_samples,))
        # return jnp.array(samples)

    def predict(self, z_new, zeigen_new):
        z_new = torch.tensor(np.array(z_new))
        zeigen_new = torch.tensor(np.array(zeigen_new))
        if z_new.ndim == 1:
            df_new = torch.stack([z_new, zeigen_new, self.X], dim=1)
            return self.predict_one_df(df_new)
        elif z_new.ndim == 2:
            n_z = z_new.shape[0]
            df_new = torch.stack([z_new, zeigen_new, self.X.repeat(n_z, 1)], dim=2)
            print("df_new shape: ", df_new.shape)
            samples_multi = []
            for i in range(n_z):
                samples_multi.append(self.predict_one_df(df_new[i]))
            return jnp.array(samples_multi).mean(axis=0)


# --- Modular bayesian inference from cut-posterior ---
# Aux functions

@jit
def compute_net_stats(Astar, Z):
    cur_eigen_cen = eigen_centrality(Astar)
    cur_Zeigen = zeigen_value(Z, cur_eigen_cen, Astar)
    # cur_ell = C*jnp.max(jnp.abs(cur_Zeigen))
    # cur_ell = jnp.array(cur_ell).reshape(1, 1)
    # return cur_Zeigen, cur_ell
    return cur_Zeigen

@jit
def get_samples_new_Astar(key, Y, Z, X, curr_Astar, true_zeigen):
# def get_samples_new_Astar(key, Y, Z, X, ell_X, curr_Astar, true_zeigen):
    # cur_Zeigen, ell_zeig = compute_net_stats(curr_Astar, Z)
    cur_Zeigen = compute_net_stats(curr_Astar, Z)
    # cur_Zeigen, _ = compute_net_stats(curr_Astar, Z)
    # ell_zeig = [C, C*jnp.max(jnp.abs(cur_Zeigen))]
    # print MAPE zeigen
    zeig_error = jnp.mean(jnp.abs(cur_Zeigen - true_zeigen)/jnp.abs(true_zeigen))
    # get df
    # cur_df = jnp.transpose(jnp.array([[1] * N, Z, X, X2, cur_Zeigen]))
    cur_df = jnp.transpose(jnp.array([[1] * N, Z, X, Z*X, cur_Zeigen]))
    # df1_gp = jnp.transpose(jnp.array([Z, X]))
    # df2_gp = jnp.transpose(jnp.array([Z, cur_Zeigen]))
    # cur_df = jnp.transpose(jnp.array([[1] * N, Z, X, cur_Zeigen]))
    # cur_df = jnp.transpose(jnp.array([[1] * N, Z, cur_Zeigen]))
    # Run MCMC
    cur_lin_samples = linear_model_samples_vectorized(key, Y, cur_df)
    # cur_gpr = Outcome_GP(X, Y, Z, cur_Zeigen, n_iter=2500, n_samples=100)
    # cur_gpr.train_model()
    # cur_hsgp_samples = HSGP_model_samples_vectorized(key, Y=Y, Xgp=cur_df[:, 2:],
    #                                        Xlin=cur_df[:, 0:2], ell=ell)
    # cur_hsgp_samples = HSGP_model_samples_vectorized(key, Y=Y,
    #                                                  df1=df1_gp, df2=df2_gp,
    #                                                     ell1=ell_X, ell2=ell_zeig)
    # cur_hsgp_samples = HSGP_model_samples_vectorized(key, Y=Y, Xgp=cur_df[:, 4:],
    #                                        Xlin=cur_df[:, 0:4], ell=ell)
    # return cur_lin_samples, cur_hsgp_samples, ell_zeig, zeig_error
    # return cur_lin_samples, cur_gpr, zeig_error
    return cur_lin_samples, cur_Zeigen, zeig_error

@jit
# def get_predicted_values(key, z, zeigen, x, x2, lin_samples, hsgp_samples,  ell_x, ell_zeig):
def get_predicted_values(key, z, zeigen, x, x2, lin_samples, gpr_model):
    # get predicted values
    # each has shape (#lin_samples, n)
    cur_lin_pred = linear_pred(z, zeigen, lin_samples, x, x2, key)
    cur_gp_pred = gpr_model.predict(z, zeigen)
    # cur_hsgp_pred = hsgp_pred(z, zeigen, hsgp_samples, x, x2, ell_x, ell_zeig)
    # get estimands for each sample (sample mean across units)
    # lin_estimates = jnp.mean(cur_lin_pred, axis=0)
    # hsgp_estimates = jnp.mean(cur_hsgp_pred, axis=0)
    # return lin_estimates, hsgp_estimates
    return cur_lin_pred, cur_gp_pred
    # return cur_lin_pred, cur_hsgp_pred


@jit
def multistage_mcmc(samp_net, Y, Z_obs, Z_h, Z_stoch, X, X2, key, true_zeigen):
    # ell_x = [C, C*jnp.max(jnp.abs(X))]
    # sample network
    curr_Astar = Triu_to_mat(samp_net)
    # re-run MCMC with new network
    curr_lin_samples, cur_Zeigen, zeig_error = get_samples_new_Astar(key, Y,
                                                         Z_obs, X,
                                                         curr_Astar, true_zeigen)
    cur_gpr = Outcome_GP(X, Y, Z_obs, cur_Zeigen, n_iter=2500, n_samples=100)
    cur_gpr.train_model()
    # curr_lin_samples, curr_hsgp_samples, ell_zeig, zeig_error = get_samples_new_Astar(key, Y,
    #                                                                                  Z_obs, X, ell_x,
    #                                                                                  curr_Astar, true_zeigen)
    # Get new zeigen values
    zeigen_h1 = compute_net_stats(curr_Astar, Z_h[0,:])
    # zeigen_h1, _ = compute_net_stats(curr_Astar, Z_h[0,:])
    # zeigen_h2, _ = compute_net_stats(curr_Astar, Z_h[1,:])
    # zeigen_stoch1, _ = compute_net_stats(curr_Astar, Z_stoch[0,:,:])
    zeigen_stoch1 = compute_net_stats(curr_Astar, Z_stoch[0,:,:])
    # zeigen_stoch2, _ = compute_net_stats(curr_Astar, Z_stoch[1,:,:])
    # get predicted estimands
    # dynamic estimands
    lin_h1, gp_h1 = get_predicted_values(key, Z_h[0,:], zeigen_h1, X, X2,
                                           curr_lin_samples, cur_gpr)
    # lin_h1, hsgp_h1 = get_predicted_values(key, Z_h[0,:], zeigen_h1, X, X2,
    #                                        curr_lin_samples, curr_hsgp_samples,
    #                                        ell_x, ell_zeig)
    # lin_h2, hsgp_h2 = get_predicted_values(key, Z_h[1,:], zeigen_h2, X, X2, curr_lin_samples,
    #                                        curr_hsgp_samples, cur_ell)
    # lin_h_estimate = lin_h1 - lin_h2
    # lin_h_estimate = jnp.mean(lin_h1 - lin_h2, axis=0)
    # lin_h_estimate = jnp.mean(lin_h1, axis=0)
    # lin_h_estimate = lin_h1
    # hsgp_h_estimate = hsgp_h1 - hsgp_h2
    # hsgp_h_estimate = jnp.mean(hsgp_h1 - hsgp_h2, axis=0)
    # hsgp_h_estimate = jnp.mean(hsgp_h1, axis=0)
    # hsgp_h_estimate = hsgp_h1

    # stochastic estimands
    lin_stoch1, gp_stoch1 = get_predicted_values(key, Z_stoch[0,:,:], zeigen_stoch1, X, X2,
                                                   curr_lin_samples, cur_gpr)
    # lin_stoch1, hsgp_stoch1 = get_predicted_values(key, Z_stoch[0,:,:], zeigen_stoch1, X, X2,
    #                                                curr_lin_samples, curr_hsgp_samples,
    #                                                ell_x, ell_zeig)
    # lin_stoch2, hsgp_stoch2 = get_predicted_values(key, Z_stoch[1,:,:], zeigen_stoch2, X, X2, curr_lin_samples,
    #                                                curr_hsgp_samples, cur_ell)
    # lin_stoch_estimate = lin_stoch1 - lin_stoch2
    # lin_stoch_estimate = lin_stoch1
    # lin_stoch_estimate = jnp.mean(lin_stoch1 - lin_stoch2, axis=0)
    # lin_stoch_estimate = jnp.mean(lin_stoch1, axis=0)
    # hsgp_stoch_estimate = hsgp_stoch1 - hsgp_stoch2
    # hsgp_stoch_estimate = hsgp_stoch1
    # hsgp_stoch_estimate = jnp.mean(hsgp_stoch1 - hsgp_stoch2, axis=0)
    # hsgp_stoch_estimate = jnp.mean(hsgp_stoch1, axis=0)
    return jnp.array([lin_h1, gp_h1, lin_stoch1, gp_stoch1]), zeig_error
    # return jnp.array([lin_h_estimate, hsgp_h_estimate, lin_stoch_estimate, hsgp_stoch_estimate]), zeig_error

parallel_multistage = pmap(multistage_mcmc, in_axes=(0, None, None, None, None, None, None, None, None))
vectorized_multistage = vmap(multistage_mcmc, in_axes=(0, None, None, None, None, None, None, None, None))


def multistage_run(multi_samp_nets, Y, Z_obs, Z_h, Z_stoch, X, X2, B, iter, h_estimand, stoch_estimand, key, true_zeigen):
    results = []
    zeig_errors = []
    for i in range(0, B, N_CORES):
        i_results = parallel_multistage(multi_samp_nets[i:(i+N_CORES),], Y,
                                        Z_obs, Z_h, Z_stoch, X, X2, key, true_zeigen)
        results.append(i_results[0])
        zeig_errors.append(i_results[1])
    results_c = jnp.concatenate(results, axis=0)
    mean_zeig_error = jnp.mean(jnp.concatenate(zeig_errors, axis=0))
    # results_c = vectorized_multistage(multi_samp_nets, Y, Z_obs, Z_new, X, key)
    n_samples = results_c.shape[2]
    # dynamic estimand
    results_lin_h = results_c[:,0,:,:]
    results_lin_h_long = results_lin_h.reshape((K*n_samples,N))
    results_gp_h = results_c[:,1,:,:]
    results_gp_h_long = results_gp_h.reshape((K*n_samples,N))
    error_stats_lin_h = compute_error_stats(esti_post_draws=results_lin_h_long,
                                          true_estimand=h_estimand,
                                          idx=iter)
    error_stats_gp_h = compute_error_stats(esti_post_draws=results_gp_h_long,
                                          true_estimand=h_estimand,
                                          idx=iter)
    # stochastic estimand
    results_lin_stoch = results_c[:,2,:,:]
    results_lin_stoch_long = results_lin_stoch.reshape((B*n_samples,N))
    results_gp_stoch = results_c[:,3,:,:]
    results_gp_stoch_long = results_gp_stoch.reshape((B*n_samples,N))
    error_stats_lin_stoch = compute_error_stats(esti_post_draws=results_lin_stoch_long,
                                               true_estimand=stoch_estimand,
                                               idx=iter)
    error_stats_gp_stoch = compute_error_stats(esti_post_draws=results_gp_stoch_long,
                                                  true_estimand=stoch_estimand,
                                                  idx=iter)
    return jnp.vstack([error_stats_lin_h, error_stats_gp_h, error_stats_lin_stoch, error_stats_gp_stoch]), mean_zeig_error
    # return jnp.vstack([error_stats_lin, error_stats_gp])
    # return pd.concat([error_stats_lin, error_stats_gp])


@jit
def network_posterior_stats(triu_sample, z):
    curr_Astar = Triu_to_mat(triu_sample)
    cur_eig_cen = eigen_centrality(curr_Astar)
    zeigen = zeigen_value(z, cur_eig_cen, curr_Astar)
    return zeigen

parallel_network_stats = pmap(network_posterior_stats, in_axes=(0, None))
vectorized_network_stats = vmap(network_posterior_stats, in_axes=(0, None))

def get_onestage_stats(multi_triu_samples, Z_obs, Z_h, Z_stoch):
    # obs_zeigen = []
    # h_zeigen = []
    # stoch_zeigen = []
    # for i in range(0, multi_triu_samples.shape[0], N_CORES):
    #     obs_results = parallel_network_stats(multi_triu_samples[i:(i + N_CORES), ], Z_obs)
    #     obs_zeigen.append(obs_results)
    #     new_results = parallel_network_stats(multi_triu_samples[i:(i + N_CORES), ], Z_h)
    #     h_zeigen.append(new_results)
    #     stoch_results = parallel_network_stats(multi_triu_samples[i:(i + N_CORES), ], Z_stoch)
    #     stoch_zeigen.append(stoch_results)
    # obs_zeigen = jnp.concatenate(obs_zeigen, axis=0)
    # h_zeigen = jnp.concatenate(h_zeigen, axis=0)
    # stoch_zeigen = jnp.concatenate(stoch_zeigen, axis=0)
    obs_zeigen = vectorized_network_stats(multi_triu_samples, Z_obs)
    h1_zeigen = vectorized_network_stats(multi_triu_samples, Z_h[0])
    h2_zeigen = vectorized_network_stats(multi_triu_samples, Z_h[1])
    stoch1_zeigen = vectorized_network_stats(multi_triu_samples, Z_stoch[0,:,:])
    stoch2_zeigen = vectorized_network_stats(multi_triu_samples, Z_stoch[1,:,:])
    mean_zeigen_obs = jnp.mean(obs_zeigen, axis=0)
    mean_zeigen_h1 = jnp.mean(h1_zeigen, axis=0)
    mean_zeigen_h2 = jnp.mean(h2_zeigen, axis=0)
    mean_zeigen_stoch1 = jnp.mean(stoch1_zeigen, axis=0)
    mean_zeigen_stoch2 = jnp.mean(stoch2_zeigen, axis=0)
    # return mean_zeigen_obs, mean_zeigen_h1, mean_zeigen_stoch1
    return mean_zeigen_obs, mean_zeigen_h1, mean_zeigen_h2, mean_zeigen_stoch1, mean_zeigen_stoch2
    # return mean_zeigen_obs, mean_zeigen_h, mean_zeigen_stoch


class Onestage_MCMC:
    def __init__(self, Y, X, X2, Z_obs, Z_h, Z_stoch, estimand_h, estimand_stoch,
                 zeigen, h1_zeigen, h2_zeigen, stoch1_zeigen, stoch2_zeigen, rng_key, iter):
        self.Y = Y
        self.X = X
        self.X2 = X2
        self.Z_obs = Z_obs
        self.n = N
        self.Z_h = Z_h
        self.Z_stoch = Z_stoch
        self.estimand_h = estimand_h
        self.estimand_stoch = estimand_stoch
        self.zeigen = zeigen
        self.h1_zeigen = h1_zeigen
        self.h2_zeigen = h2_zeigen
        self.stoch1_zeigen = stoch1_zeigen
        self.stoch2_zeigen = stoch2_zeigen
        self.rng_key = rng_key
        self.iter = iter
        self.df_lin = self.get_df()
        # self.ell = jnp.array(C*jnp.max(jnp.abs(self.df[:,4]))).reshape(1,1)
        # self.ell1 = [C, C * jnp.max(jnp.abs(self.X))]
        # self.ell2 = [C, C * jnp.max(jnp.abs(self.zeigen))]
        # self.df1 = jnp.transpose(jnp.array([self.Z_obs, self.X]))
        # self.df2 = jnp.transpose(jnp.array([self.Z_obs, self.zeigen]))
        self.linear_post_samples = linear_model_samples_parallel(key=self.rng_key, Y=self.Y, df=self.df_lin)
        self.gp_reg = Outcome_GP(self.X, self.Y, self.Z_obs, self.zeigen)
        self.gp_reg.train_model()
        # self.hsgp_post_samples = HSGP_model_samples_parallel(key=self.rng_key, Y=self.Y,
        #                                                     df1=self.df1, df2=self.df2,
        #                                                     ell1=self.ell1, ell2=self.ell2)

    def get_df(self):
        # return jnp.transpose(jnp.array([[1]*self.n, self.Z_obs, self.zeigen]))
        return jnp.transpose(jnp.array([[1]*self.n, self.Z_obs, self.X, self.Z_obs*self.X, self.zeigen]))
        # return jnp.transpose(jnp.array([[1]*self.n, self.Z_obs, self.X, self.zeigen]))
        # return jnp.transpose(jnp.array([[1]*self.n, self.Z_obs, self.X, self.X2, self.zeigen]))

    def get_results(self):
        # dynamic (h) intervention
        linear_h1_pred = linear_pred(self.Z_h[0,:], self.h1_zeigen, self.linear_post_samples,
                                    self.X, self.X2, self.rng_key)
        # linear_h2_pred = linear_pred(self.Z_h[1,:], self.h2_zeigen, self.linear_post_samples,
        #                              self.X, self.X2, self.rng_key)
        # linear_h = jnp.mean(linear_h1_pred, axis=0) - jnp.mean(linear_h2_pred, axis=0)
        # linear_h = linear_h1_pred - linear_h2_pred
        linear_h = linear_h1_pred
        # linear_h = jnp.mean(linear_h1_pred - linear_h2_pred, axis=0)
        # linear_h = jnp.mean(linear_h1_pred, axis=0)
        linear_h_stats = compute_error_stats(linear_h, self.estimand_h, idx=self.iter)

        # hsgp_h1_pred = hsgp_pred(self.Z_h[0,:], self.h1_zeigen, self.hsgp_post_samples,
        #                         self.X, self.X2, self.ell1, self.ell2)
        # # hsgp_h2_pred = hsgp_pred(self.Z_h[1,:], self.h2_zeigen, self.hsgp_post_samples,
        # #                          self.X, self.X2, self.ell, self.rng_key)
        # # hsgp_h = jnp.mean(hsgp_h1_pred, axis=0) - jnp.mean(hsgp_h2_pred, axis=0)
        # # hsgp_h = hsgp_h1_pred - hsgp_h2_pred
        # hsgp_h = hsgp_h1_pred
        # # hsgp_h = jnp.mean(hsgp_h1_pred - hsgp_h2_pred, axis=0)
        # # hsgp_h = jnp.mean(hsgp_h1_pred, axis=0)
        # hsgp_h_stats = compute_error_stats(hsgp_h, self.estimand_h, idx=self.iter)
        gp_h_pred = self.gp_reg.predict(self.Z_h[0, :], self.h1_zeigen)
        gp_h_stats = compute_error_stats(gp_h_pred, self.estimand_h, idx=self.iter)


        # stochastic intervention
        linear_stoch_pred1 = linear_pred(self.Z_stoch[0,:,:], self.stoch1_zeigen,
                                        self.linear_post_samples, self.X, self.X2, self.rng_key)
        # linear_stoch_pred2 = linear_pred(self.Z_stoch[1,:,:], self.stoch2_zeigen,
        #                                  self.linear_post_samples, self.X, self.X2, self.rng_key)
        # linear_stoch_pred = jnp.mean(linear_stoch_pred1, axis=0) - jnp.mean(linear_stoch_pred2, axis=0)
        # linear_stoch_pred = linear_stoch_pred1 - linear_stoch_pred2
        linear_stoch_pred = linear_stoch_pred1
        # linear_stoch_pred = jnp.mean(linear_stoch_pred1 - linear_stoch_pred2, axis=0)
        # linear_stoch_pred = jnp.mean(linear_stoch_pred1, axis=0)
        linear_stoch_stats = compute_error_stats(linear_stoch_pred, self.estimand_stoch, idx=self.iter)

        # hsgp_stoch_pred1 = hsgp_pred(self.Z_stoch[0,:,:], self.stoch1_zeigen, self.hsgp_post_samples,
        #                             self.X, self.X2, self.ell1, self.ell2)
        # # hsgp_stoch_pred2 = hsgp_pred(self.Z_stoch[1,:,:], self.stoch2_zeigen, self.hsgp_post_samples,
        # #                              self.X, self.X2, self.ell, self.rng_key)
        # # hsgp_stoch_pred = jnp.mean(hsgp_stoch_pred1, axis=0) - jnp.mean(hsgp_stoch_pred2, axis=0)
        # # hsgp_stoch_pred = hsgp_stoch_pred1 - hsgp_stoch_pred2
        # hsgp_stoch_pred = hsgp_stoch_pred1
        # # hsgp_stoch_pred = jnp.mean(hsgp_stoch_pred1 - hsgp_stoch_pred2, axis=0)
        # # hsgp_stoch_pred = jnp.mean(hsgp_stoch_pred1, axis=0)
        # hsgp_stoch_stats = compute_error_stats(hsgp_stoch_pred,
        #                                    self.estimand_stoch, idx=self.iter)
        gp_stoch_pred = self.gp_reg.predict(self.Z_stoch[0, :, :], self.stoch1_zeigen)
        gp_stoch_stats = compute_error_stats(gp_stoch_pred,
                                             self.estimand_stoch, idx=self.iter)

        return jnp.vstack([linear_h_stats, gp_h_stats, linear_stoch_stats, gp_stoch_stats])
        # return jnp.vstack([linear_h_stats, hsgp_h_stats, linear_stoch_stats, hsgp_stoch_stats])
        # return pd.concat([linear_h_stats, hsgp_h_stats, linear_stoch_stats, hsgp_stoch_stats])
