# Load libraries
import numpy as np
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
from hsgp.approximation import hsgp_squared_exponential, eigenfunctions
from hsgp.spectral_densities import diag_spectral_density_squared_exponential
import pyro
import pyro.contrib.gp as gp
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from joblib import Parallel, delayed


# --- Set cores and seed ---
N_CORES = 8
# N_CORES = 20
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_CORES}"

# --- Set global variables and values ---
N = 500
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

    def generate_X(self, loc=0, scale=3):
        return jnp.array(self.rng.normal(loc=loc, scale=scale, size=self.n))

    def generate_X2(self,p=0.1):
        return jnp.array(self.rng.binomial(n=1, p=p, size=self.n))

    def generate_U_latent(self, loc=0, scale=1, K=2):
        return jnp.array(self.rng.normal(loc=loc, scale=scale, size=(self.n, K)))

    def generate_Z(self, p):
        return jnp.array(self.rng.binomial(n=1, p=p, size=self.n))

    def x_diff(self):
        idx_pairs = list(combinations(range(self.n), 2))
        x_d = jnp.array([abs(self.X[i] - self.X[j]) for i, j in idx_pairs])
        return x_d

    def x2_equal(self):
        idx_pairs = list(combinations(range(self.n), 2))
        x2_equal = jnp.array([1 if (self.X2[i] + self.X2[j] == 1) else 0 for i, j in idx_pairs])
        return x2_equal

    def latent_to_norm_of_diff(self):
        idx = jnp.triu_indices(n=self.n, k=1)
        U_diff = self.U_latent[idx[0]] - self.U_latent[idx[1]]
        return jnp.linalg.norm(U_diff, axis=1)

    def generate_triu(self):
        probs = expit(self.theta[0] + self.theta[1]*self.X2_equal - self.U_diff_norm)
        return self.rng.binomial(n=1, p=probs, size=self.triu_dim)

    def generate_adj_matrix(self):
        mat = np.zeros((self.n, self.n))
        idx_upper_tri = np.triu_indices(n=self.n, k=1)
        mat[idx_upper_tri] = self.triu
        return mat + mat.T

    def gen_outcome(self, z, zeig, with_epsi):
        df_lin = jnp.transpose(np.array([[1]*self.n, z, self.X, self.X2]))
        if self.lin:
            mean_y = jnp.dot(jnp.column_stack((df_lin, zeig)), self.eta)
        else:
            return "Non-linear not implemented"
        if with_epsi:
            epsi = jnp.array(self.rng.normal(loc=0, scale=self.sig_y, size=self.n))
            Y = jnp.array(mean_y + epsi)
            return Y, epsi
        else:
            return mean_y

    def dynamic_intervention(self, thresholds=(1.5, 2)):
        Z_h1 = jnp.where((self.X > thresholds[0]) | (self.X < -thresholds[0]), 1, 0)
        Z_h2 = jnp.where((self.X > thresholds[1]) | (self.X < -thresholds[1]), 1, 0)
        return jnp.array([Z_h1, Z_h2])

    def stochastic_intervention(self, n_approx=1000):
        z_stoch1 = self.rng.binomial(n=1, p=self.alphas[0], size=(n_approx, self.n))
        z_stoch2 = self.rng.binomial(n=1, p=self.alphas[1], size=(n_approx, self.n))
        return jnp.array([z_stoch1, z_stoch2])

    def get_true_estimand(self, z_new):
        if z_new.ndim == 3:
            zeigen_new1 = zeigen_value(z_new[0,:,:], self.eig_cen, self.adj_mat)
            zeigen_new2 = zeigen_value(z_new[1,:,:], self.eig_cen, self.adj_mat)
            n_stoch = z_new.shape[1]
            results = np.zeros((n_stoch, N))
            for i in range(n_stoch):
                y1 = self.gen_outcome(z_new[0,i,], zeigen_new1[i,], with_epsi=False)
                y2 = self.gen_outcome(z_new[1,i,], zeigen_new2[i,], with_epsi=False)
                results[i,] = y1 - y2
            return jnp.mean(results, axis=0).squeeze()
        else:
            zeigen_new1 = zeigen_value(z_new[0,:], self.eig_cen, self.adj_mat)
            zeigen_new2 = zeigen_value(z_new[1,:], self.eig_cen, self.adj_mat)
            y1 = self.gen_outcome(z_new[0,], zeigen_new1, with_epsi=False)
            y2 = self.gen_outcome(z_new[1,], zeigen_new2, with_epsi=False)
            return y1 - y2

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
def create_noisy_network(rng, triu_vals, gamma, x_diff, x2_eq):
    obs_mat = np.zeros((N, N))  # create nXn matrix of zeros
    triu_idx = np.triu_indices(n=N, k=1)
    logit_nois = triu_vals*gamma[0] + (1-triu_vals)*(gamma[1] + gamma[2]*x_diff + gamma[3]*x2_eq)
    edges_noisy = rng.binomial(n=1, p=expit(logit_nois), size=TRIL_DIM)
    obs_mat[triu_idx] = edges_noisy
    obs_mat = obs_mat + obs_mat.T
    return {"obs_mat" : obs_mat,
            "triu_obs" : edges_noisy}

@jit
def Triu_to_mat(triu_v):
    adj_mat = jnp.zeros((N,N))
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


def compute_error_stats(esti_post_draws, true_estimand, idx=None):
    # esti_post_draws has shape (M, N)
    # true_estimand has shape (N,)
    mean_estimand = jnp.mean(true_estimand)  # scalar
    mean_units = jnp.mean(esti_post_draws, axis=1)  # shape (M,)
    mean_samples = jnp.mean(esti_post_draws, axis=0)  # shape (N,)
    mean_all = jnp.round(jnp.mean(esti_post_draws), 3)  # scalar
    medi = jnp.round(jnp.median(mean_units),3)
    std = jnp.round(jnp.std(mean_units),3)
    RMSE_all = jnp.round(jnp.sqrt(jnp.mean(jnp.power(esti_post_draws - true_estimand, 2))),3)
    RMSE = jnp.round(jnp.sqrt(jnp.mean(jnp.power(mean_units - mean_estimand, 2))),3)
    MAE_all = jnp.round(jnp.mean(jnp.abs(esti_post_draws - true_estimand)), 3)
    MAE = jnp.round(jnp.mean(jnp.abs(mean_units - mean_estimand)),3)
    MAPE = jnp.round(jnp.mean(jnp.abs((mean_units - mean_estimand) / (mean_estimand + 1e-5))), 3)
    MAPE_all = jnp.round(jnp.mean(jnp.abs((esti_post_draws - true_estimand[None, :]) / (true_estimand[None, :] + 1e-5))), 3)
    rel_RMSE = jnp.round(jnp.mean(jnp.square((mean_units - mean_estimand) / (mean_estimand + 1e-5))), 3)
    rel_RMSE_all = jnp.round(jnp.mean(jnp.square((esti_post_draws - true_estimand[None, :]) / (true_estimand[None, :] + 1e-5))), 3)
    q025 = jnp.quantile(mean_units, 0.025)
    q025_ind = jnp.quantile(esti_post_draws, 0.025, axis=0)
    q975 = jnp.quantile(mean_units, 0.975)
    q975_ind = jnp.quantile(esti_post_draws, 0.975, axis=0)
    # hdi_lower, hdi_upper = compute_hdi(esti_post_draws, 0.95)
    cover = (q025 <= mean_estimand) & (mean_estimand <= q975)
    mean_cover = (q025_ind <= true_estimand) & (true_estimand <= q975_ind)
    return jnp.array([idx, mean_all, medi, jnp.round(mean_estimand,3),
                      jnp.round(mean_all - mean_estimand,3), std,
                      RMSE,RMSE_all, MAE,  MAE_all,
                      MAPE, MAPE_all, rel_RMSE, rel_RMSE_all,
                      jnp.round(q025,3), jnp.round(q975,3),
                      cover, jnp.mean(mean_cover)])

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
    with pyro.plate("Latent_dim", N):
        nu = pyro.sample("nu",
                         pyro.distributions.MultivariateNormal(torch.zeros(K) + eps, torch.eye(K)))

    idx = torch.triu_indices(N, N, offset=1)
    nu_diff = nu[idx[0]] - nu[idx[1]]
    nu_diff_norm_val = torch.norm(nu_diff, dim=1)

    with pyro.plate("theta_dim", 2):
        theta = pyro.sample("theta",
                            pyro.distributions.Normal(0, 5))

    mu_net = theta[0] + x2 * theta[1] - nu_diff_norm_val
    mu_net = torch.clamp(mu_net, min=-30, max=30)

    with pyro.plate("gamma_i", 4):
        gamma = pyro.sample("gamma",
                            pyro.distributions.Normal(0, 5))

    with pyro.plate("A* and A", x.shape[0]):
        triu_star = pyro.sample("triu_star",
                                pyro.distributions.Bernoulli(logits=mu_net),
                                infer={"enumerate": "parallel"})

        logit_misspec = torch.where(triu_star == 1.0,
                                    gamma[0],
                                    gamma[1] + gamma[2] * x + gamma[3]*x2)

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
    sig = numpyro.sample("sig", dist.LogNormal(scale=2.0))
    mu_y = jnp.dot(df, eta)
    # --- likelihood --
    with numpyro.plate("obs", df.shape[0]):
        numpyro.sample("Y", dist.Normal(loc=mu_y, scale=sig), obs=Y)


def HSGP_model(df, ell, m, Y=None):
    # --- Priors ---
    amplitude1 = numpyro.sample("amplitude1", dist.HalfNormal(2))
    length1 = numpyro.sample("lengthscale1", dist.HalfNormal(5))
    noise = numpyro.sample("noise", dist.HalfNormal(2))
    # --- GP ---
    f1 = numpyro.deterministic(
        "f1_star",
        hsgp_squared_exponential(
            x=df[:,1:], alpha=amplitude1, length=length1, ell=ell, m=m, i="1"
            # df1, alpha=amplitude1, length=length1, ell=ell1, m=m, i="1"
        ),
    )
    # --- Likelihood ---
    intercept = numpyro.sample("eta_0",dist.Normal(0,5))
    x_eta = numpyro.sample("eta_1", dist.Normal(0,5))
    f = intercept + x_eta*df[:,0] + f1
    numpyro.sample("Y", dist.Normal(f, noise), obs=Y)


# --- MCMC aux functions ---
# @jit
def linear_model_samples_parallel(key, Y, df):
    kernel_outcome = NUTS(outcome_model)
    lin_mcmc = MCMC(kernel_outcome, num_warmup=2000, num_samples=2500,num_chains=4, progress_bar=True)
    lin_mcmc.run(key, df=df, Y=Y)
    return lin_mcmc.get_samples()

@jit
def linear_model_samples_vectorized(key, Y, df):
    kernel_outcome = NUTS(outcome_model)
    lin_mcmc = MCMC(kernel_outcome, num_warmup=1000, num_samples=250, num_chains=1,
                    progress_bar=False, chain_method="vectorized")
    lin_mcmc.run(key, df=df, Y=Y)
    return lin_mcmc.get_samples()


@jit
def outcome_jit_pred(post_samples, df_arr, key):
    pred_func = Predictive(outcome_model, post_samples)
    return pred_func(key, df_arr)

# @jit
def HSGP_model_samples_parallel(key, Y, df, ell):
    kernel_hsgp = NUTS(HSGP_model, target_accept_prob=0.9)
    hsgp_mcmc = MCMC(kernel_hsgp, num_warmup=4000, num_samples=2500,num_chains=4, progress_bar=True)
    hsgp_mcmc.run(key, df=df, ell=ell, m=M, Y=Y)
    return hsgp_mcmc.get_samples()

@jit
def HSGP_model_samples_vectorized(key, Y, df, ell):
    kernel_hsgp = NUTS(HSGP_model, target_accept_prob=0.9)
    hsgp_mcmc = MCMC(kernel_hsgp, num_warmup=2000, num_samples=250,num_chains=1,
                     progress_bar=False, chain_method="vectorized")
    hsgp_mcmc.run(key, df=df, ell=ell, m=M, Y=Y)
    return hsgp_mcmc.get_samples()

@jit
def HSGP_jit_pred(post_samples, df, ell, key):
    pred_func = Predictive(HSGP_model, post_samples)
    return pred_func(key, df=df, ell=ell, m=M)

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

def manual_gp_f_star_pred(df, ell, post_samples):
    f_hat = compute_f_star(df[:,1:], ell,
                         post_samples["amplitude1"],
                         post_samples["lengthscale1"],
                         post_samples["beta1"])
    intercept = jnp.array(post_samples["eta_0"])
    ones = jnp.ones(N)
    x_eta = jnp.array(post_samples["eta_1"])
    return intercept[:, jnp.newaxis]*ones[jnp.newaxis,:] + x_eta[:,jnp.newaxis]*df[:,0][jnp.newaxis,:] + jnp.array(f_hat)

@jit
def linear_model_outcome_pred(z, zeigen, post_samples, x, x2, key):
    df = jnp.transpose(jnp.array([[1] * N, z, x, zeigen]))
    # pred = outcome_jit_pred(post_samples, df, key)
    # return jnp.mean(pred["Y"], axis=1)
    return manual_linear_pred(post_samples, df)

linear_model_pred = vmap(linear_model_outcome_pred, in_axes=(0, 0, None, None, None, None))

def linear_pred(z, zeigen, post_samples, x, x2, key):
    if z.ndim == 2:
        return jnp.mean(linear_model_pred(z, zeigen, post_samples, x, x2, key), axis=0)
    if z.ndim == 1:
        n_z = z.shape[0]
        return linear_model_pred(z.reshape((1, n_z)), zeigen.reshape((1, n_z)), post_samples, x, x2, key).squeeze()

@jit
def hsgp_model_outcome_pred(z, zeigen, post_samples, x, x2, ell, key):
    # ell_ = jnp.array(c*jnp.max(jnp.abs(zeigen))).reshape(1,1)
    df = jnp.transpose(jnp.array([x, z, zeigen]))
    # return HSGP_jit_pred(post_samples, df, ell, key)["Y"]
    return manual_gp_f_star_pred(df, ell, post_samples)

hsgp_model_pred = vmap(hsgp_model_outcome_pred, in_axes=(0, 0, None, None, None, None, None))

def hsgp_pred(z, zeigen, post_samples, x, x2, ell, key):
    if z.ndim == 2:
        return jnp.mean(hsgp_model_pred(z, zeigen, post_samples, x, x2, ell, key), axis=0)
    if z.ndim == 1:
        n_z = z.shape[0]
        return hsgp_model_pred(z.reshape((1, n_z)), zeigen.reshape((1, n_z)), post_samples, x, x2, ell, key).squeeze()


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
    def __init__(self, data, rng_key, n_iter=20000, n_samples=10000, network_model=pyro_noisy_networks_model):
        self.X_diff = torch.tensor(np.array(data["X_diff"]), dtype=torch.float32)
        self.X2_eq = torch.tensor(np.array(data["X2_equal"]), dtype=torch.float32)
        self.triu = torch.tensor(np.array(data["triu"]), dtype=torch.float32)
        self.n = N
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


class Network_MCMC:
    def __init__(self, data, rng_key, n_warmup=4000, n_samples=4000, n_chains=4):
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
        self.ell = [C, C*jnp.max(jnp.abs(self.zeigen))]
        self.df_lin = self.get_df()
        self.df_gp = jnp.transpose(jnp.array([self.X, self.Z, self.zeigen]))
        self.Z_h = data["Z_h"]
        self.Z_stoch = data["Z_stoch"]
        self.estimand_h = data["estimand_h"]
        self.estimand_stoch = data["estimand_stoch"]
        self.type = type
        self.rng_key = rng_key
        self.iter = iter
        self.linear_post_samples = linear_model_samples_parallel(key=self.rng_key, Y=self.Y, df=self.df_lin)
        self.hsgp_post_samples = HSGP_model_samples_parallel(key=self.rng_key, Y=self.Y,
                                                             df=self.df_gp, ell=self.ell)
        self.print_zeig_error(data)

    def print_zeig_error(self, data):
        if self.type == "observed":
            obs_mae_error = jnp.mean(jnp.abs(self.zeigen - data["Zeigen"]))
            print("Obs. mape zeigen: ",obs_mae_error)

    def get_df(self):
        return jnp.transpose(jnp.array([[1]*self.n, self.Z, self.X, self.zeigen]))

    def get_results(self):
        # dynamic (h) intervention
        h1_zeigen = zeigen_value(self.Z_h[0,:], self.eig_cen, self.adj_mat)
        h2_zeigen = zeigen_value(self.Z_h[1,:], self.eig_cen, self.adj_mat)

        linear_h1_pred = linear_pred(self.Z_h[0,:], h1_zeigen, self.linear_post_samples,
                                    self.X, self.X2, self.rng_key)
        linear_h2_pred = linear_pred(self.Z_h[1,:], h2_zeigen, self.linear_post_samples,
                                      self.X, self.X2, self.rng_key)
        linear_h = linear_h1_pred - linear_h2_pred
        linear_h_stats = compute_error_stats(linear_h, self.estimand_h, idx=self.iter)

        hsgp_h1_pred = hsgp_pred(self.Z_h[0,:], h1_zeigen, self.hsgp_post_samples,
                                self.X, self.X2, self.ell, self.rng_key)
        hsgp_h2_pred = hsgp_pred(self.Z_h[1,:], h2_zeigen, self.hsgp_post_samples,
                                self.X, self.X2, self.ell, self.rng_key)
        hsgp_h = hsgp_h1_pred - hsgp_h2_pred
        hsgp_h_stats = compute_error_stats(hsgp_h, self.estimand_h, idx=self.iter)

        # stochastic intervention
        stoch_zeigen1 = zeigen_value(self.Z_stoch[0,:,:], self.eig_cen, self.adj_mat)
        stoch_zeigen2 = zeigen_value(self.Z_stoch[1,:,:], self.eig_cen, self.adj_mat)

        linear_stoch_pred1 = linear_pred(self.Z_stoch[0,:,:], stoch_zeigen1,
                                        self.linear_post_samples, self.X, self.X2, self.rng_key)
        linear_stoch_pred2 = linear_pred(self.Z_stoch[1,:,:], stoch_zeigen2,
                                         self.linear_post_samples, self.X, self.X2, self.rng_key)
        linear_stoch_pred = linear_stoch_pred1 - linear_stoch_pred2
        linear_stoch_stats = compute_error_stats(linear_stoch_pred, self.estimand_stoch, idx=self.iter)

        hsgp_stoch_pred1 = hsgp_pred(self.Z_stoch[0,:,:], stoch_zeigen1, self.hsgp_post_samples,
                                    self.X, self.X2, self.ell, self.rng_key)
        hsgp_stoch_pred2 = hsgp_pred(self.Z_stoch[1,:,:], stoch_zeigen2, self.hsgp_post_samples,
                                    self.X, self.X2, self.ell, self.rng_key)
        hsgp_stoch_pred = hsgp_stoch_pred1 - hsgp_stoch_pred2
        hsgp_stoch_stats = compute_error_stats(hsgp_stoch_pred,
                                           self.estimand_stoch, idx=self.iter)

        return jnp.vstack([linear_h_stats, hsgp_h_stats, linear_stoch_stats, hsgp_stoch_stats])


def robust_cholesky(matrix, max_tries=5, initial_jitter=1e-6):
    jitter = initial_jitter
    num_tries = 0
    while num_tries < max_tries:
        try:
            L = torch.linalg.cholesky(matrix + torch.eye(matrix.shape[0]) * jitter)
            return L
        except RuntimeError:
            jitter *= 10
            num_tries += 1
    raise ValueError(f"Matrix is not positive definite, even with jitter of {jitter}")

class Outcome_GP:
    def __init__(self, X, Y, Z, Zeigen, n_iter = 5000, n_samples = 10000):
        self.X = torch.from_numpy(np.array(X))
        self.Y = torch.from_numpy(np.array(Y))
        self.Z = torch.from_numpy(np.array(Z))
        self.zeigen = torch.from_numpy(np.array(Zeigen))
        self.n = N
        self.df = self.get_df()
        self.gpr = self.gpr_model()
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.dataset = TensorDataset(self.df, self.Y)
        self.batch_size = self.n // 2  # Adjust based on your dataset size and available memory
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.post_samples = None

    def get_df(self):
        return torch.stack([self.Z, self.zeigen, self.X], dim=1)

    def gpr_model(self):
        kernel = gp.kernels.RBF(input_dim=3, variance=torch.tensor(2.0), lengthscale=torch.tensor(3.0))
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
        for _ in range(self.n_iter):
            for batch_X, batch_y in self.dataloader:
                optimizer.zero_grad()
                # Update the model's data for this batch
                self.gpr.set_data(batch_X, batch_y)
                loss = loss_fn(self.gpr.model, self.gpr.guide)
                loss.backward()
                optimizer.step()

    def predict_one_df(self, df):
        with torch.no_grad():
            # mean, cov = self.gpr(df, full_cov=True, noiseless=False)
            mean, cov = self.gpr(df, full_cov=True, noiseless=True)
            L = robust_cholesky(cov)
            # Generate samples from standard normal distribution
            eps = torch.randn(cov.shape[0], self.n_samples)
            # Transform to samples from multivariate normal
            samples = mean.unsqueeze(1) + L @ eps
        return jnp.array(samples.T)

    def predict(self, z_new, zeigen_new):
        z_new = torch.tensor(np.array(z_new))
        zeigen_new = torch.tensor(np.array(zeigen_new))
        if z_new.ndim == 1:
            df_new = torch.stack([z_new, zeigen_new, self.X], dim=1)
            return self.predict_one_df(df_new)
        elif z_new.ndim == 2:
            n_z = z_new.shape[0]
            df_new = torch.stack([z_new, zeigen_new, self.X.repeat(n_z, 1)], dim=2)
            samples_multi = []
            for i in range(n_z):
                samples_multi.append(self.predict_one_df(df_new[i]))
            return jnp.array(samples_multi).mean(axis=0)


def train_and_predict_single_gpr(args):
    x, y, z_obs, zeigen_m, z_h, z_stoch, zeigen_h, zeigen_stoch, n_iter, n_samples = args
    # Train the model
    gpr = Outcome_GP(x, y, z_obs, zeigen_m, n_iter=n_iter, n_samples=n_samples)
    gpr.train_model()
    # Make prediction
    gp_h_pred = gpr.predict(z_h, zeigen_h)
    # gp_stoch_pred = gpr.predict(z_stoch, zeigen_stoch)
    return jnp.array([gp_h_pred])
    # return jnp.array([gp_h_pred, gp_stoch_pred])


def run_parallel_gpr(args_list, num_processes=N_CORES):
    with tqdm(total=len(args_list), desc="GP multiple (parallel)") as pbar:
        results = Parallel(n_jobs=num_processes, backend="loky")(
            delayed(train_and_predict_single_gpr)(args) for args in args_list
        )
        pbar.update(len(args_list))

    return jnp.array(results)


# --- Modular bayesian inference from cut-posterior ---
# Aux functions


def get_args_list(x, y, z_obs, zeigen_df, z_h, z_stoch, zeigen_h, zeigen_stoch, n_iter=1500, n_samples=250):
    return [(x, y, z_obs, zeigen_m, z_h, z_stoch, zeig_h, zeig_stoch, n_iter, n_samples)
                     for zeigen_m, zeig_h, zeig_stoch in zip(zeigen_df, zeigen_h, zeigen_stoch)]

def GP_multistage(x, y, z_obs, zeigen_df, z_h, z_stoch, zeigen_h, zeigen_stoch, h_estimand, stoch_estimand, iter, n_iter=1500, n_samples=250):
    args_list = get_args_list(x, y, z_obs, zeigen_df, z_h[0,:], z_stoch[0,:,:], zeigen_h, zeigen_stoch, n_iter, n_samples)
    gp_run = run_parallel_gpr(args_list)
    gp_h_pred = gp_run
    gp_h_pred_long = gp_h_pred.reshape(-1, gp_h_pred.shape[-1])
    gp_h_error_stats = compute_error_stats(gp_h_pred_long, h_estimand, idx=iter)
    return jnp.vstack([gp_h_error_stats])


@jit
def compute_net_stats(Astar, Z):
    cur_eigen_cen = eigen_centrality(Astar)
    cur_Zeigen = zeigen_value(Z, cur_eigen_cen, Astar)
    return cur_Zeigen

@jit
def get_samples_new_Astar(key, Y, Z, X, ell_X, curr_Astar, true_zeigen):
    cur_Zeigen = compute_net_stats(curr_Astar, Z)
    ell_zeig = [C, C*jnp.max(jnp.abs(cur_Zeigen))]
    # print MAPE zeigen
    zeig_error = jnp.mean(jnp.abs(cur_Zeigen - true_zeigen)/jnp.abs(true_zeigen))
    # get df
    cur_df = jnp.transpose(jnp.array([[1] * N, Z, X, Z*X, cur_Zeigen]))
    df1_gp = jnp.transpose(jnp.array([Z, X]))
    df2_gp = jnp.transpose(jnp.array([Z, cur_Zeigen]))
    # Run MCMC
    cur_lin_samples = linear_model_samples_vectorized(key, Y, cur_df)
    cur_hsgp_samples = HSGP_model_samples_vectorized(key, Y=Y,
                                                     df1=df1_gp, df2=df2_gp,
                                                        ell1=ell_X, ell2=ell_zeig)
    return cur_lin_samples, cur_hsgp_samples, ell_zeig, zeig_error

@jit
def get_predicted_values(key, z, zeigen, x, x2, lin_samples, hsgp_samples,  ell):
    # each has shape (#lin_samples, n)
    cur_lin_pred = linear_pred(z, zeigen, lin_samples, x, x2, key)
    cur_hsgp_pred = hsgp_pred(z, zeigen, hsgp_samples, x, x2, ell, key)
    return cur_lin_pred, cur_hsgp_pred


@jit
def single_stage_run(zeigen, zeigen_h1, zeigen_h2, zeigen_stoch, zeigen_stoch2, x, x2, y, z_obs, z_h, z_stoch, key):
    # get samples from linear outcome model
    df_lin = jnp.transpose(jnp.array([[1] * N, z_obs, x, zeigen]))
    lin_samples = linear_model_samples_vectorized(key, y, df_lin)
    # get prediction for z_h1 and z_stoch
    lin_h1_pred = linear_pred(z_h[0,:], zeigen_h1, lin_samples, x, x2, key)
    lin_h2_pred = linear_pred(z_h[1,:], zeigen_h2, lin_samples, x, x2, key)
    lin_h_pred = lin_h1_pred - lin_h2_pred

    lin_stoch1_pred = linear_pred(z_stoch[0,:,:], zeigen_stoch, lin_samples, x, x2, key)
    lin_stoch2_pred = linear_pred(z_stoch[1,:,:], zeigen_stoch2, lin_samples, x, x2, key)
    lin_stoch_pred = lin_stoch1_pred - lin_stoch2_pred

    # Repeat for HSGP model
    df_gp = jnp.transpose(jnp.array([x, z_obs, zeigen]))
    ell = [C, C*jnp.max(jnp.abs(zeigen))]
    hsgp_samples = HSGP_model_samples_vectorized(key, Y=y, df=df_gp, ell=ell)
    # Get HSGP predictions
    hsgp_h1_pred = hsgp_pred(z_h[0,:], zeigen_h1, hsgp_samples, x, x2, ell, key)
    hsgp_h2_pred = hsgp_pred(z_h[1,:], zeigen_h2, hsgp_samples, x, x2, ell, key)
    hsgp_h_pred = hsgp_h1_pred - hsgp_h2_pred

    hsgp_stoch1_pred = hsgp_pred(z_stoch[0,:,:], zeigen_stoch, hsgp_samples, x, x2, ell, key)
    hsgp_stoch2_pred = hsgp_pred(z_stoch[1,:,:], zeigen_stoch2, hsgp_samples, x, x2, ell, key)
    hsgp_stoch_pred = hsgp_stoch1_pred - hsgp_stoch2_pred

    # return predictions
    return jnp.array([lin_h_pred, lin_stoch_pred, hsgp_h_pred, hsgp_stoch_pred])

parallel_stage_run = pmap(single_stage_run, in_axes=(0,0,0,0,0, None, None, None, None, None, None, None))


def multistage_run(zeigen_post, zeigen_h1_post, zeigen_h2_post, zeigen_stoch_post, zeigen_stoch2_post,
                   x, x2, y, z_obs, z_h, z_stoch, h_estimand, stoch_estimand, iter, key):
    B = zeigen_post.shape[0]
    results = []
    for i in tqdm(range(0, B, N_CORES), desc="Mutlistage run"):
        i_results = parallel_stage_run(zeigen_post[i:(i + N_CORES), ],
                                        zeigen_h1_post[i:(i + N_CORES), ],
                                        zeigen_h2_post[i:(i + N_CORES), ],
                                        zeigen_stoch_post[i:(i + N_CORES), ],
                                        zeigen_stoch2_post[i:(i + N_CORES), ],
                                        x, x2, y, z_obs, z_h, z_stoch, key)
        results.append(i_results)
    results_c = jnp.concatenate(results, axis=0)
    n_samples = results_c.shape[2]

    # save error stats for linear models
    results_lin_h = results_c[:, 0, :, :]
    results_lin_h_long = results_lin_h.reshape((B * n_samples, N))
    error_stats_lin_h = compute_error_stats(esti_post_draws=results_lin_h_long,
                                            true_estimand=h_estimand,
                                            idx=iter)
    results_lin_stoch = results_c[:, 1, :, :]
    results_lin_stoch_long = results_lin_stoch.reshape((B * n_samples, N))
    error_stats_lin_stoch = compute_error_stats(esti_post_draws=results_lin_stoch_long,
                                               true_estimand=stoch_estimand,
                                               idx=iter)
    # error stats for HSGP
    results_hsgp_h = results_c[:, 2, :, :]
    results_hsgp_h_long = results_hsgp_h.reshape((B * n_samples, N))
    error_stats_hsgp_h = compute_error_stats(esti_post_draws=results_hsgp_h_long,
                                            true_estimand=h_estimand,
                                            idx=iter)
    results_hsgp_stoch = results_c[:, 3, :, :]
    results_hsgp_stoch_long = results_hsgp_stoch.reshape((B * n_samples, N))
    error_stats_hsgp_stoch = compute_error_stats(esti_post_draws=results_hsgp_stoch_long,
                                               true_estimand=stoch_estimand,
                                               idx=iter)
    # return results
    return jnp.vstack([error_stats_lin_h, error_stats_lin_stoch, error_stats_hsgp_h, error_stats_hsgp_stoch])


@jit
def network_posterior_stats(triu_sample, z):
    curr_Astar = Triu_to_mat(triu_sample)
    cur_eig_cen = eigen_centrality(curr_Astar)
    zeigen = zeigen_value(z, cur_eig_cen, curr_Astar)
    return zeigen

parallel_network_stats = pmap(network_posterior_stats, in_axes=(0, None))
vectorized_network_stats = vmap(network_posterior_stats, in_axes=(0, None))

def get_post_net_stats(multi_triu_samples, Z_obs, Z_h, Z_stoch):
    obs_zeigen = vectorized_network_stats(multi_triu_samples, Z_obs)
    h1_zeigen = vectorized_network_stats(multi_triu_samples, Z_h[0])
    h2_zeigen = vectorized_network_stats(multi_triu_samples, Z_h[1])
    stoch1_zeigen = vectorized_network_stats(multi_triu_samples, Z_stoch[0,:,:])
    stoch2_zeigen = vectorized_network_stats(multi_triu_samples, Z_stoch[1,:,:])
    return obs_zeigen, h1_zeigen, h2_zeigen, stoch1_zeigen, stoch2_zeigen

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
        self.ell = [C, C*jnp.max(jnp.abs(self.zeigen))]
        self.df_gp = jnp.transpose(jnp.array([self.X, self.Z_obs, self.zeigen]))
        self.linear_post_samples = linear_model_samples_parallel(key=self.rng_key, Y=self.Y, df=self.df_lin)
        self.hsgp_post_samples = HSGP_model_samples_parallel(key=self.rng_key, Y=self.Y,
                                                            df=self.df_gp, ell=self.ell)

    def get_df(self):
        return jnp.transpose(jnp.array([[1]*self.n, self.Z_obs, self.X, self.zeigen]))

    def get_results(self):
        # dynamic (h) intervention
        linear_h1_pred = linear_pred(self.Z_h[0,:], self.h1_zeigen, self.linear_post_samples,
                                    self.X, self.X2, self.rng_key)
        linear_h2_pred = linear_pred(self.Z_h[1,:], self.h2_zeigen, self.linear_post_samples,
                                     self.X, self.X2, self.rng_key)
        linear_h = linear_h1_pred - linear_h2_pred
        linear_h_stats = compute_error_stats(linear_h, self.estimand_h, idx=self.iter)

        hsgp_h1_pred = hsgp_pred(self.Z_h[0,:], self.h1_zeigen, self.hsgp_post_samples,
                                self.X, self.X2, self.ell, self.rng_key)
        hsgp_h2_pred = hsgp_pred(self.Z_h[1,:], self.h2_zeigen, self.hsgp_post_samples,
                                 self.X, self.X2, self.ell, self.rng_key)
        hsgp_h = hsgp_h1_pred - hsgp_h2_pred
        hsgp_h_stats = compute_error_stats(hsgp_h, self.estimand_h, idx=self.iter)

        # stochastic intervention
        linear_stoch_pred1 = linear_pred(self.Z_stoch[0,:,:], self.stoch1_zeigen,
                                        self.linear_post_samples, self.X, self.X2, self.rng_key)
        linear_stoch_pred2 = linear_pred(self.Z_stoch[1,:,:], self.stoch2_zeigen,
                                         self.linear_post_samples, self.X, self.X2, self.rng_key)
        linear_stoch_pred = linear_stoch_pred1 - linear_stoch_pred2
        linear_stoch_stats = compute_error_stats(linear_stoch_pred, self.estimand_stoch, idx=self.iter)

        hsgp_stoch_pred1 = hsgp_pred(self.Z_stoch[0,:,:], self.stoch1_zeigen, self.hsgp_post_samples,
                                    self.X, self.X2, self.ell, self.rng_key)
        hsgp_stoch_pred2 = hsgp_pred(self.Z_stoch[1,:,:], self.stoch2_zeigen, self.hsgp_post_samples,
                                     self.X, self.X2, self.ell, self.rng_key)
        hsgp_stoch_pred = hsgp_stoch_pred1 - hsgp_stoch_pred2
        hsgp_stoch_stats = compute_error_stats(hsgp_stoch_pred,
                                           self.estimand_stoch, idx=self.iter)
        return jnp.vstack([linear_h_stats, hsgp_h_stats, linear_stoch_stats, hsgp_stoch_stats])
