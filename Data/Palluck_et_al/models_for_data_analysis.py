import jax.lax
import numpy as np
import pandas as pd
import os
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
from jax.scipy.special import expit, logit
import numpyro.distributions as dist
import numpyro
from numpyro.contrib.funsor import config_enumerate
import pyro
import torch
# import
from numpyro.infer import MCMC, NUTS, Predictive
from hsgp.approximation import hsgp_squared_exponential, eigenfunctions
from hsgp.spectral_densities import diag_spectral_density_squared_exponential

# --- Utility functions ---


# --- Network models ---

# Either one or two noisy networks
# In both, A* is created from
# \nu_i \in R^K
# P(A*_ij|X_i, X_j, \theta, \nu_i, \nu_j) = expit(\theta_intercept + \theta*I(X_i = X_j) + \theta_latent||\nu_i - \nu_j||)
# A1 (ST W1) from:
# P(A_ij|A*_ij = 1,X,\gamma) = expit(\gamma_0)
# P(A_ij|A*_ij = 0,X,\gamma) = expit(\gamma_1)
# if relevant, A2 (ST W2) from:
# P(A_ij|A*_ij = 1, A1_ij ,X,\gamma) = expit(\gamma_2*(I(A1_ij = 1) +  \gamma_3*(I(A1_ij = 0))
# P(A_ij|A*_ij = 0, A1_ij ,X,\gamma) = expit(\gamma_4*(I(A1_ij = 1) +  \gamma_5*(I(A1_ij = 0))


@pyro.infer.config_enumerate
def one_noisy_networks_model(x_df, triu_v, N, K=2, eps=1e-3):
    """
    Network model for one noisy observed network. True network is geenrated from LSM.
    :param x_df: pairwise x-differences/equality
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

    theta_intercept = pyro.sample("theta_intercept", pyro.distributions.Normal(0, 5))
    with pyro.plate("theta_dim", x_df.shape[1]):
        theta = pyro.sample("theta",
                            pyro.distributions.Normal(0, 5))

    mu_net = theta_intercept + torch.matmul(x_df, theta) - nu_diff_norm_val
    # mu_net = theta_intercept + x_df @ theta - nu_diff_norm_val
    mu_net = torch.clamp(mu_net, min=-30, max=30)
    with pyro.plate("gamma_i", 2 + x_df.shape[1]):
        gamma = pyro.sample("gamma",
                            pyro.distributions.Normal(0, 5))

    with pyro.plate("A* and A", x_df.shape[0]):
        triu_star = pyro.sample("triu_star",
                                pyro.distributions.Bernoulli(logits=mu_net),
                                infer={"enumerate": "parallel"})

        logit_misspec = torch.where(triu_star == 1.0,
                                    gamma[0],
                                    gamma[1] + x_df @ gamma[2:])

        pyro.sample("obs_triu",
                    pyro.distributions.Bernoulli(logits=logit_misspec),
                    obs=triu_v)

@pyro.infer.config_enumerate
def repeated_noisy_networks_model(x_df, triu_v, N, K=2, eps=1e-3):
    """
    Network model for one noisy observed network. True network is geenrated from LSM.
    :param x_df: pairwise x-differences/equality
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

    theta_intercept = pyro.sample("theta_intercept", pyro.distributions.Normal(0, 5))
    with pyro.plate("theta_dim", x_df.shape[1]):
        theta = pyro.sample("theta",
                            pyro.distributions.Normal(0, 5))

    mu_net = theta_intercept + torch.matmul(x_df, theta) - nu_diff_norm_val
    # mu_net = theta_intercept + x_df @ theta - nu_diff_norm_val
    mu_net = torch.clamp(mu_net, min=-30, max=30)
    with pyro.plate("gamma_A1", 2):
        gamma_a1 = pyro.sample("gamma_1",
                               pyro.distributions.Normal(0, 5))

    with pyro.plate("gamma_A2", 4):
        gamma_a2 = pyro.sample("gamma_2",
                               pyro.distributions.Normal(0, 5))

    with pyro.plate("A* and A", x_df.shape[0]):
        triu_star = pyro.sample("triu_star",
                                pyro.distributions.Bernoulli(logits=mu_net),
                                infer={"enumerate": "parallel"})

        logit_misspec_A1 = torch.where(triu_star == 1.0,
                                       gamma_a1[0],
                                       gamma_a1[1])

        logit_misspec_A2 = torch.where(triu_star == 1.0,
                                       gamma_a2[0] + gamma_a2[1] * triu_v[0, :],
                                       gamma_a2[2] + gamma_a2[3] * triu_v[0, :])

        pyro.sample("obs_triu_A1",
                    pyro.distributions.Bernoulli(logits=logit_misspec_A1),
                    obs=triu_v[0, :] if triu_v is not None else None)
        pyro.sample("obs_triu_A2",
                    pyro.distributions.Bernoulli(logits=logit_misspec_A2),
                    obs=triu_v[1, :] if triu_v is not None else None)

@pyro.infer.config_enumerate
def multilayer_networks_model(x_df, triu_v, N, K=2, eps=1e-3):
    """
    Network model for one noisy observed network. True network is geenrated from LSM.
    :param x_df: pairwise x-differences/equality
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

    theta_intercept = pyro.sample("theta_intercept", pyro.distributions.Normal(0, 5))
    with pyro.plate("theta_dim", x_df.shape[1]):
        theta = pyro.sample("theta",
                            pyro.distributions.Normal(0, 5))

    mu_net = theta_intercept + torch.matmul(x_df, theta) - nu_diff_norm_val
    # mu_net = theta_intercept + x_df @ theta - nu_diff_norm_val
    mu_net = torch.clamp(mu_net, min=-30, max=30)

    with pyro.plate("gamma_A1", 2 + x_df.shape[1]):
        gamma_a1 = pyro.sample("gamma_1",
                               pyro.distributions.Normal(0, 5))

    with pyro.plate("gamma_A2", 2 + x_df.shape[1]):
        gamma_a2 = pyro.sample("gamma_2",
                               pyro.distributions.Normal(0, 5))

    with pyro.plate("A* and A", x_df.shape[0]):
        triu_star = pyro.sample("triu_star",
                                pyro.distributions.Bernoulli(logits=mu_net),
                                infer={"enumerate": "parallel"})

        logit_misspec_A1 = torch.where(triu_star == 1.0,
                                       gamma_a1[0],
                                       gamma_a1[1] + x_df @ gamma_a1[2:])
        # logit_misspec_A1 = gamma_a1[0] + gamma_a1[1] * triu_star + torch.matmul(x_df, gamma_a1[2:])
        # logit_misspec_A2 = gamma_a2[0] + gamma_a2[1] * triu_star + torch.matmul(x_df, gamma_a2[2:])
        logit_misspec_A2 = torch.where(triu_star == 1.0,
                                       gamma_a2[0],
                                       gamma_a2[1] + x_df @ gamma_a2[2:])

        pyro.sample("obs_triu_A1",
                    pyro.distributions.Bernoulli(logits=logit_misspec_A1),
                    obs=triu_v[0, :] if triu_v is not None else None)
        pyro.sample("obs_triu_A2",
                    pyro.distributions.Bernoulli(logits=logit_misspec_A2),
                    obs=triu_v[1, :] if triu_v is not None else None)


@config_enumerate
# def noisy_networks_model(x_eq: jnp.ndarray, triu_v: jnp.ndarray, N_edges: int, N: int, K = 2):
def noisy_networks_model(x_eq: jnp.ndarray, triu_v: jnp.ndarray, N: int, K = 2):
    # True network priors
    # Latent variable for each unit from bi-normal distribution
    # sigma_sq = numpyro.sample("sigma_sq", dist.InverseGamma(0.1, 1.0))
    # cov_matrix = sigma_sq * jnp.eye(K)
    # cov_matrix = 5.0 * jnp.eye(K)
    # N = ((1 + jnp.sqrt(1 + 8*x_eq.shape[0]))/2).astype(int)
    # N = jax.lax.convert_element_type((1 + jnp.sqrt(1 + 8*x_eq.shape[0]))/2, jnp.int32)
    # with numpyro.plate("nu_i", N):
        # nu = numpyro.sample("nu", dist.MultivariateNormal(loc=jnp.zeros(K), covariance_matrix=cov_matrix))
        # nu_standard = numpyro.sample("nu_standard", dist.Normal(jnp.zeros(K), jnp.ones(K)))
        # nu_standard = numpyro.sample("nu_standard", dist.MultivariateNormal(loc=jnp.zeros(K), covariance_matrix=jnp.eye(K)))
        # nu = numpyro.deterministic("nu", nu_standard*jnp.sqrt(sigma_sq))
        # nu = numpyro.deterministic("nu", nu_standard*3.0)

    # nu_standard = numpyro.sample("nu_standard", dist.Normal(0, 1).expand((N, K)))

    # nu = numpyro.deterministic("nu", nu_standard * jnp.sqrt(sigma_sq))
    # Calculate pairwise differences
    # idx = jnp.triu_indices(n=N, k=1)
    # nu_diff = nu[idx[0]] - nu[idx[1]]
    # nu_diff_norm_val = jnp.linalg.norm(nu_diff, axis=1)
    # nu_diff = nu[:, None, :] - nu[None, :, :]  # Shape: (N*(N-1)/2, N*(N-1)/2, K)
    # nu_diff_norm = jnp.linalg.norm(nu_diff, axis=2)  # Shape: (N*(N-1)/2, N*(N-1)/2)
    # nu_diff_norm_val = nu_diff_norm[jnp.triu_indices(n=N,k=1)] # Shape: (N*(N-1)/2,)
    # theta_latent = numpyro.sample("theta_latent", dist.Normal(0, 5))

    # Priors for covariates (fixed effects).
    theta_intercept = numpyro.sample("theta_intercept", dist.Normal(0, 5))
    with numpyro.plate("theta_i", x_eq.shape[1]):
        theta = numpyro.sample("theta", dist.Normal(0, 5))
    # theta_intercept = numpyro.sample("theta_intercept", dist.Normal(0, 5))
    # theta = numpyro.sample("theta", dist.Normal(0, 5).expand((x_eq.shape[1],)))
    # Save logits for A*
    # mu_net = theta_intercept + jnp.dot(x_eq, theta) + theta_latent*nu_diff_norm_val
    # mu_net = theta_intercept + jnp.dot(x_eq, theta) - nu_diff_norm_val
    mu_net = theta_intercept + jnp.dot(x_eq, theta)
    # mu_net = theta_intercept + jnp.dot(x_eq, theta)

    if triu_v.ndim == 1:
        with numpyro.plate("gamma_i", 2 + x_eq.shape[1]):
        # # with numpyro.plate("gamma_i", 2):
            gamma = numpyro.sample("gamma", dist.Normal(0, 5))
        # gamma = numpyro.sample("gamma", dist.Normal(0, 5).expand((3,)))

        with numpyro.plate("A* and A", x_eq.shape[0]):
            triu_star = numpyro.sample("triu_star", dist.Bernoulli(logits=mu_net)
                                       # , infer={"enumerate": "sequential"})
                                       , infer={"enumerate": "parallel"})
            # logit_misspec = triu_star*gamma[0] + (1 - triu_star)*gamma[1]
            # logit_misspec = triu_star*gamma[0] + (1 - triu_star)*(gamma[1] + gamma[2]*nu_diff_norm_val)
            logit_misspec = jnp.where(triu_star,
                                      gamma[0],
                                      gamma[1] + jnp.dot(x_eq, gamma[2:]))
                                      # gamma[1] + gamma[2] * nu_diff_norm_val)
            numpyro.sample("obs_triu", dist.Bernoulli(logits=logit_misspec), obs=triu_v)

    else: # triu_v.ndim == 2
        with numpyro.plate("gamma_A1", 3):
        # with numpyro.plate("gamma_A1", 2):
            gamma_a1 = numpyro.sample("gamma_1", dist.Normal(0, 5))

        with numpyro.plate("gamma_A2", 5):
        # with numpyro.plate("gamma_A2", 4):
            gamma_a2 = numpyro.sample("gamma_2", dist.Normal(0, 5))

        with ((numpyro.plate("A* and A", x_eq.shape[0]))):
            triu_star = numpyro.sample("triu_star", dist.Bernoulli(logits=mu_net),
                                       infer={"enumerate": "parallel"})
            # logit_misspec_A1 = triu_star*gamma_a1[0] + (1 - triu_star)*gamma_a1[1]
            # logit_misspec_A1 = triu_star*gamma_a1[0] + (1 - triu_star)*(gamma_a1[1] + gamma_a1[2]*nu_diff_norm_val)
            logit_misspec_A1 = jnp.where(triu_star,
                                         gamma_a1[0],
                                         gamma_a1[1] )
                                         # gamma_a1[1] + gamma_a1[2] * nu_diff_norm_val)
            # logit_misspec_A2 = triu_star*(gamma_a2[0] + gamma_a2[1]*triu_v[0,:])
            # + (1 - triu_star)*(gamma_a2[2] + gamma_a2[3]*triu_v[0,:] + gamma_a2[4]*nu_diff_norm_val)
            logit_misspec_A2 = jnp.where(triu_star,
                                         gamma_a2[0] + gamma_a2[1]*triu_v[0,:],
                                         gamma_a2[2] + gamma_a2[3]*triu_v[0,:])
                                         # gamma_a2[2] + gamma_a2[3]*triu_v[0,:] + gamma_a2[4]*nu_diff_norm_val)
            # logit_misspec_A2 = triu_star*(gamma_a2[0]*(1 - triu_v[0,:]) + gamma_a2[1]*triu_v[0,:])
            # + (1 - triu_star)*((gamma_a2[2] + gamma_a2[3]*nu_diff_norm_val)*(1 - triu_v[0,:]) + (gamma_a2[4] + gamma_a2[5]*nu_diff_norm_val)*triu_v[0,:])
            # + (1 - triu_star)*(gamma_a2[2]*(1 - triu_v[0,:]) + gamma_a2[3]*triu_v[0,:])

            numpyro.sample("obs_triu_A1", dist.Bernoulli(logits=logit_misspec_A1), obs=triu_v[0,:])
            numpyro.sample("obs_triu_A2", dist.Bernoulli(logits=logit_misspec_A2), obs=triu_v[1,:])

def outcome_model(df, Y=None):
    # --- priors ---
    with numpyro.plate("Lin coef.", df.shape[1]):
        eta = numpyro.sample("eta", dist.Normal(0, 5))
    # sig = numpyro.sample("sig", dist.HalfNormal(scale=2))
    # sig = numpyro.sample("sig", dist.LogNormal(scale=2.0))
    mu_y = df @ eta
    # --- likelihood --
    with numpyro.plate("obs", df.shape[0]):
        numpyro.sample("Y", dist.Bernoulli(logits=mu_y), obs=Y)
