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

# --- Utility functions ---

def n_edges_to_n_nodes(n_edges):
    return int((1 + jnp.sqrt(1 + 8*n_edges))/2)

class Network_SVI:
    def __init__(self, x_df, triu_obs, n_iter=15000, n_samples=10000, network_model=models.one_noisy_networks_model):
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
        for _ in tqdm(range(self.n_iter), desc="Training network model"):
            svi.step(self.x_df, self.triu_obs, self.N)
            # loss = svi.step(self.X_diff, self.X2_eq, self.triu, self.n)
            # losses_full.append(loss)

    def network_samples(self):
        triu_star_samples = []
        for _ in tqdm(range(self.n_samples), desc="Sampling A*"):
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

        print(posterior_samples.keys())

        # Extract triu_obs samples
        if self.triu_obs.ndim == 1:
            triu_obs_samples = posterior_samples['obs_triu']
        else:
            triu_obs_A1 = posterior_samples['obs_triu_A1']
            triu_obs_A2 = posterior_samples['obs_triu_A2']
            triu_obs_samples = torch.stack([triu_obs_A1, triu_obs_A2])
        return triu_obs_samples


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