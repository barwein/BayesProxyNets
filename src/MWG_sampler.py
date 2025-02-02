###
# Metrpolis-within-Gibbs sampler with HMCGibbs
# GWG is used for discrete parameters (triu_star)
# Initalization of parameters from the cut-posterior
###

import jax
import jax.numpy as jnp
from jax import random, vmap
from numpyro.infer import SVI, TraceGraph_ELBO, MCMC, NUTS, HMC, Predictive
from numpyro.infer.hmc_gibbs import HMCGibbs
from numpyro.optim import ClippedAdam
from numpyro.infer.autoguide import AutoDelta

import src.Models as models
import src.utils as utils
from src.GWG import make_gwg_gibbs_fn

import os

# --- Set cores and seed ---
N_CORES = 4
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_CORES}"


# Init values from the cut-posterior


def sample_posterior_triu_star(key, probs, num_samples=2000):
    """
    Sample Bernoulli realizations for each probability vector in probs

    Args:
        key: JAX PRNGKey
        probs: array of shape (n_samples, M) with probabilities

    Returns:
        array of shape (n_samples, M) with Bernoulli samples
    """
    # Split key for each sample
    keys = random.split(key, num_samples)

    # Define single sample function
    def sample_single(key):
        return jnp.astype(random.bernoulli(key, probs), jnp.int32)

    # Vectorize over samples and probs
    vectorized_sample = vmap(sample_single)

    return vectorized_sample(keys)


def replicate_params(init_params, num_chains):
    """
    Replicate initial parameters for multiple chains
    """
    return jax.tree.map(lambda x: jnp.repeat(x[None], num_chains, axis=0), init_params)


class MWG_init:
    def __init__(
        self,
        rng_key,
        data,
        cut_posterior_net_model=models.network_only_models_marginalized,
        cut_posterior_outcome_model=models.plugin_outcome_model,
        triu_star_log_posterior_fn=models.compute_log_posterior_vmap,
        n_iter_networks=10000,
        n_nets_samples=2000,
        n_warmup_outcome=2000,
        n_samples_outcome=2500,
        learning_rate=0.01,
    ):
        self.rng_key = rng_key
        self.data = data
        self.cut_posterior_net_model = cut_posterior_net_model
        self.cut_posterior_outcome_model = cut_posterior_outcome_model
        self.triu_star_log_posterior_fn = triu_star_log_posterior_fn
        self.n_iter_networks = n_iter_networks
        self.n_nets_samples = n_nets_samples
        self.n_warmup_outcome = n_warmup_outcome
        self.n_samples_outcome = n_samples_outcome
        self.learning_rate = learning_rate

        # initial parameters
        self.theta = None
        self.gamma = None
        self.triu_star_probs = None
        self.triu_star = None
        self.exposures = None
        self.eta = None
        self.rho = None
        self.sig_inv = None

    def init_network_params(self):
        """
        Initialize network parameters from cut-posterior
        """
        runkey, sampkey = random.split(self.rng_key)

        # Define guide
        guide = AutoDelta(self.cut_posterior_net_model)

        # Define SVI
        svi = SVI(
            model=self.cut_posterior_net_model,
            guide=guide,
            optim=ClippedAdam(self.learning_rate),
            loss=TraceGraph_ELBO(),
        )

        # Run SVI
        svi_results = svi.run(runkey, self.n_iter_networks, self.data)

        # Get MAP of theta and gamma
        map_params = guide.median(svi_results.params)

        self.theta = map_params["theta"]
        self.gamma = map_params["gamma"]

        # get A* posterior probs
        preds = Predictive(
            self.cut_posterior_net_model, guide=guide, params=svi_results.params
        )
        predictions = preds(sampkey, self.data)
        self.triu_star_probs = predictions["triu_star_probs"][0]

    def init_triu_star_and_exposures(self):
        """
        Initialize triu_star and exposures using theta and gamma samples
        """
        triu_star_samps = sample_posterior_triu_star(
            self.rng_key, self.triu_star_probs, self.n_nets_samples
        )

        # get exposures --> init is the posterior mean exposures
        exposures_samps = utils.vmap_compute_exposures(triu_star_samps, self.data.Z)
        self.exposures = exposures_samps.mean(axis=0)

        # get triu_star --> init is the posterior triu_star with largest log-posterior density
        triu_star_log_post = self.triu_star_log_posterior_fn(
            triu_star_samps, self.theta, self.gamma, self.data
        )
        best_idx = jnp.argmax(triu_star_log_post)
        self.triu_star = triu_star_samps[best_idx]

    def init_outcome_model(self):
        """
        Initialize outcome model parameters from cut-posterior
        """
        df_nodes = jnp.transpose(
            jnp.stack(
                [
                    jnp.ones(self.data.Y.shape[0]),
                    self.data.Z,
                    self.data.x,
                    self.exposures,
                ]
            )
        )

        # Define MCMC kernel and run MCMC
        kernel_plugin = NUTS(self.cut_posterior_outcome_model)
        mcmc_plugin = MCMC(
            kernel_plugin,
            num_warmup=self.n_warmup_outcome,
            num_samples=self.n_samples_outcome,
            num_chains=4,
            progress_bar=False,
        )
        mcmc_plugin.run(
            self.rng_key, df_nodes, utils.Triu_to_mat(self.triu_star), self.data.Y
        )

        # Get posterior samples
        samples = mcmc_plugin.get_samples()

        self.eta = samples["eta"].mean(axis=0)
        self.rho = samples["rho"].mean()
        self.sig_inv = samples["sig_inv"].mean()

    def get_init_values(self, num_chains=4):
        """
        Get initial values for the MWG sampler

        Args:
            num_chains: int, number of chains

        Returns:
            dict with initial values for the MWG sampler
        """

        print("Initializing parameters for MWG sampler...")

        self.init_network_params()
        self.init_triu_star_and_exposures()
        self.init_outcome_model()
        init_params = {
            "theta": self.theta,
            "gamma": self.gamma,
            "triu_star": self.triu_star,
            "eta": self.eta,
            "rho": self.rho,
            "sig_inv": self.sig_inv,
        }
        if num_chains > 1:
            return replicate_params(init_params, num_chains)
        else:
            return init_params


# MWG sampler


class MWG_sampler:
    def __init__(
        self,
        rng_key,
        data,
        init_params,
        combined_model=models.combined_model,
        n_warmup=2000,
        n_samples=2500,
        num_chains=4,
        continuous_sampler="NUTS",  # one of "NUTS" or "HMC"
        progress_bar=True,
    ):
        self.rng_key = rng_key
        self.data = data
        self.init_params = init_params
        self.combined_model = combined_model
        self.continuous_sampler = continuous_sampler

        self.gwg_fn = make_gwg_gibbs_fn(data)
        self.continuous_kernel = self.make_continuous_kernel()

        self.mwg_kernel = HMCGibbs(
            inner_kernel=self.continuous_kernel,
            gibbs_fn=self.gwg_fn,
            gibbs_sites=["triu_star"],
        )

        self.mwg_mcmc = MCMC(
            self.mwg_kernel,
            num_warmup=n_warmup,
            num_samples=n_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )

        self.posterior_samples = None

    def make_continuous_kernel(self):
        if self.continuous_sampler == "NUTS":
            return NUTS(self.combined_model)
        elif self.continuous_sampler == "HMC":
            return HMC(self.combined_model)
        else:
            raise ValueError("continuous_sampler must be one of 'NUTS' or 'HMC'")

    def run(self):
        self.mwg_mcmc.run(self.rng_key, self.data, init_params=self.init_params)
        self.posterior_samples = self.mwg_mcmc.get_samples()

    def get_posterior_samples(self):
        return self.posterior_samples

    # TODO: add Predictive function that reutrn predictive distribution of Y (outcome)
