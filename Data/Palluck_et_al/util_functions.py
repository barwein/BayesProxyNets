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
from numpyro.infer import MCMC, NUTS, Predictive
from hsgp.approximation import hsgp_squared_exponential, eigenfunctions
from hsgp.spectral_densities import diag_spectral_density_squared_exponential
from numpyro_models import noisy_networks_model

# --- Utility functions ---

def n_edges_to_n_nodes(n_edges):
    return int((1 + jnp.sqrt(1 + 8*n_edges))/2)

class Network_MCMC:
    # def __init__(self, data, rng_key, n_warmup=1000, n_samples=1500, n_chains=4):
    def __init__(self, network_model, x_eq, triu_v, rng_key, n_warmup=2000, n_samples=4000, n_chains=4, K=2):
    # def __init__(self, data, n, rng_key, n_warmup=1000, n_samples=2000, n_chains=6):
        self.network_model = network_model
        self.x_eq = x_eq
        self.triu_v = triu_v
        self.N_edges = self.x_eq.shape[0]
        self.N = n_edges_to_n_nodes(self.N_edges)
        self.rng_key = rng_key
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.K = K
        self.network_m = self.network()
        self.post_samples = None

    def network(self):
        kernel = NUTS(self.network_model,
                      target_accept_prob=0.9,
                      init_strategy=numpyro.infer.init_to_median(num_samples=20)
                      )
        return MCMC(kernel, num_warmup=self.n_warmup, num_samples=self.n_samples,
                    num_chains=self.n_chains, progress_bar=True)
                    # num_chains=self.n_chains, progress_bar=False)

    def get_posterior_samples(self):
        self.network_m.run(self.rng_key,
                           x_eq=self.x_eq,
                           triu_v=self.triu_v,
                           # N_edges = self.N_edges,
                           N = self.N,
                           K = self.K)
        self.network_m.print_summary()
        self.post_samples = self.network_m.get_samples()
        return self.post_samples

    def predictive_samples(self):
        posterior_predictive = Predictive(model=self.network_model,
                                          posterior_samples=self.post_samples,
                                          infer_discrete=True)
        A_star_pred = posterior_predictive(self.rng_key,
                                          x_eq=self.x_eq,
                                          triu_v=self.triu_v,
                                          K = self.K)["triu_star"]
        nu_pred = posterior_predictive(self.rng_key,
                                          x_eq=self.x_eq,
                                          triu_v=self.triu_v,
                                          K = self.K)["nu"]
        return A_star_pred, nu_pred
