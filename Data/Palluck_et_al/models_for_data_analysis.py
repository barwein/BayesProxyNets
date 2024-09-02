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

# --- Global variables ---

NUM_SCHOOLS = 1
# NUM_SCHOOLS = 56
NUM_GRADES = 3

# --- Network models ---

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


def outcome_model(trts, exposures, sch_treat, fixed_df, grade, school, Y=None):

    # --- fixed effects priors ---
    with numpyro.plate("Fixed effects", fixed_df.shape[1]):
        eta = numpyro.sample("eta", dist.Normal(0, 5))
    fixed_effects = jnp.dot(fixed_df, eta)

    # --- School random effect ---
    mu_sch = numpyro.sample('mu_sch', dist.Normal(0, 5))
    sigma_sch = numpyro.sample('sigma_sch', dist.LogNormal(0, 2))
    with numpyro.plate("School", NUM_SCHOOLS):
        eta_sch_std = numpyro.sample("eta_sch_std", dist.Normal(0, 1))
    eta_sch = mu_sch + sigma_sch * eta_sch_std

    # --- Grade random effect ---
    mu_grade = numpyro.sample('mu_grade', dist.Normal(0, 5))
    sigma_grade = numpyro.sample('sigma_grade', dist.LogNormal(0, 2))
    with numpyro.plate("Grade", NUM_GRADES):
        eta_grade_std = numpyro.sample("eta_grade_std", dist.Normal(0, 1))
    eta_grade = mu_grade + sigma_grade * eta_grade_std

    # --- treatment effect ---
    eta_trt = numpyro.sample("eta_trt", dist.Normal(0, 5))
    eta_exposures = numpyro.sample("eta_exposures", dist.Normal(0, 5))
    eta_interaction = numpyro.sample("eta_interaction", dist.Normal(0, 5))
    treat_effect = sch_treat*(eta_trt*trts + eta_exposures*exposures + eta_interaction*trts*exposures)

    # --- outcome model ---
    mu_y = fixed_effects + eta_sch[school] + eta_grade[grade] + treat_effect
    # --- likelihood --
    with numpyro.plate("obs", fixed_df.shape[0]):
        numpyro.sample("Y", dist.Bernoulli(logits=mu_y),
                       obs=Y if Y is not None else None)
