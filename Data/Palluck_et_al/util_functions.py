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
from hsgp.approximation import hsgp_squared_exponential, eigenfunctions
from hsgp.spectral_densities import diag_spectral_density_squared_exponential
from numpyro_models import noisy_networks_model

# --- Utility functions ---

def n_edges_to_n_nodes(n_edges):
    return int((1 + jnp.sqrt(1 + 8*n_edges))/2)

class Network_MCMC:
    # def __init__(self, data, rng_key, n_warmup=1000, n_samples=1500, n_chains=4):
    def __init__(self, network_model, x_eq, triu_v, rng_key, n_warmup=2000, n_samples=4000, n_chains=4, K=2, MCMC = True):
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
        self.MCMC = MCMC
        self.network_m = self.network()
        self.post_samples = None
        self.guide = None

    def network(self):
        if self.MCMC:
            kernel = NUTS(self.network_model,
                          target_accept_prob=0.9,
                          init_strategy=numpyro.infer.init_to_median(num_samples=20)
                          )
            return MCMC(kernel, num_warmup=self.n_warmup, num_samples=self.n_samples,
                        num_chains=self.n_chains, progress_bar=True)
                        # num_chains=self.n_chains, progress_bar=False)
        else:
            # self.guide = self.guide()
            self.guide = AutoNormal(numpyro.handlers.block(self.network_model, hide=["triu_star"]))
            # self.guide = AutoNormal(self.network_model)
            # self.guide = AutoNormal(numpyro.handlers.substitute(noisy_networks_model, lambda site: None if site["name"] == "triu_star" else site["value"]))
            # self.guide = numpyro.infer.autoguide.AutoGuide(noisy_networks_model)
            # svi = SVI(self.network_model, self.guide, numpyro.optim.Adam(step_size=0.005), TraceEnum_ELBO())
            svi = SVI(self.network_model, self.guide, numpyro.optim.Adam(step_size=0.005), TraceEnum_ELBO())
            # svi = SVI(self.network_model, self.guide, numpyro.optim.Adam(step_size=0.005), TraceGraph_ELBO())
            return svi

    # def guide(self):
    #     # We use `handlers.substitute` to avoid sampling 'triu_star' in the guide
    #     with numpyro.handlers.substitute(numpyro.handlers.block(self.network_model, hide=["triu_star"])):
    #         return AutoNormal(self.network_model)(self.x_eq, self.triu_v, self.N, self.K)

    def run_svi(self):
        # svi_state = self.network_m.init(self.rng_key, self.x_eq, self.triu_v, self.N, self.K)
        # num_steps = 10000
        # for step in range(num_steps):
        #     svi_state, loss = self.network_m.update(svi_state, self.x_eq, self.triu_v, self.N, self.K)
        #     if step % 100 == 0:
        #         print(f'Step {step} Loss: {loss}')
        # return self.network_m.get_params(svi_state)
        svi_results = self.network_m.run(self.rng_key,num_steps=10000,
                                         x_eq=self.x_eq, triu_v=self.triu_v, N=self.N, K=self.K)
        return svi_results.params

    def get_posterior_samples(self):
        if self.MCMC:
            self.network_m.run(self.rng_key,
                               x_eq=self.x_eq,
                               triu_v=self.triu_v,
                               # N_edges = self.N_edges,
                               N = self.N,
                               K = self.K)
            self.network_m.print_summary()
            self.post_samples = self.network_m.get_samples()
            return self.post_samples
        else:
            self.post_samples = self.run_svi()
            return self.post_samples

    def predictive_samples(self):
        if self.MCMC:
            posterior_predictive = Predictive(model=self.network_model,
                                              posterior_samples=self.post_samples,
                                              infer_discrete=True)

            return posterior_predictive(self.rng_key,
                                        x_eq=self.x_eq,
                                        triu_v=self.triu_v,
                                        N=self.N,
                                        K=self.K)
        else:
            predictive = Predictive(self.guide, params=self.post_samples, num_samples=1000)
            return predictive(self.rng_key, x_eq=self.x_eq, triu_v=self.triu_v, N=self.N, K=self.K)