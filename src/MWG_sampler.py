###
# Metrpolis-within-Gibbs sampler with HMCGibbs
# GWG is used for discrete parameters (triu_star)
# Initalization of parameters from the cut-posterior
###

import jax
import jax.numpy as jnp
from jax import random, vmap
from numpyro.infer import MCMC, NUTS, HMC, Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.optim import ClippedAdam, Adam
from numpyro.infer.hmc_gibbs import HMCGibbs
import numpyro.handlers as handlers

import src.Models as models
import src.utils as utils
from src.GWG import make_gwg_gibbs_fn, make_gwg_gibbs_fn_rep


# Init values from the cut-posterior


def sample_posterior_triu_star(key, probs, num_samples=2000):
    """
    Sample Bernoulli realizations for each probability vector in probs

    Args:
        key: JAX PRNGKey
        probs: array of posterior probabilities (N(N-1)/2)

    Returns:
        array of shape (n_samples, N(N-1)/2) with Bernoulli samples
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


def get_gamma_length(model, data):
    # Use a dummy random key
    rng_key = jax.random.PRNGKey(0)
    # Run the model under a trace handler to record sample sites.
    tr = handlers.trace(handlers.seed(model, rng_key)).get_trace(data)
    return tr["gamma"]["value"].shape[0]


class MWG_init:
    def __init__(
        self,
        rng_key,
        data,
        cut_posterior_net_model=models.networks_marginalized_model,
        cut_posterior_outcome_model=models.plugin_outcome_model,
        triu_star_log_posterior_fn=models.compute_log_posterior_vmap,
        n_iter_networks=20000,
        n_nets_samples=3000,
        n_warmup_outcome=2000,
        n_samples_outcome=2500,
        learning_rate=0.0005,
        num_chains_outcome=4,
        progress_bar=False,
    ):
        self.rng_key = random.split(rng_key)[0]
        self.data = data
        self.cut_posterior_net_model = cut_posterior_net_model
        self.cut_posterior_outcome_model = cut_posterior_outcome_model
        self.triu_star_log_posterior_fn = triu_star_log_posterior_fn
        self.n_iter_networks = n_iter_networks
        self.n_nets_samples = n_nets_samples
        self.n_warmup_outcome = n_warmup_outcome
        self.n_samples_outcome = n_samples_outcome
        self.learning_rate = learning_rate
        self.num_chains_outcome = num_chains_outcome
        self.progress_bar = progress_bar

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
        gamma_len = get_gamma_length(self.cut_posterior_net_model, self.data)
        init_vals = {
            "theta": jnp.zeros(2),
            "gamma": jnp.zeros(gamma_len),
        }

        # init with SVI with AutoMultivariateNormal guide
        guide = AutoMultivariateNormal(self.cut_posterior_net_model)

        optimizer = ClippedAdam(self.learning_rate)
        # optimizer = Adam(self.learning_rate)

        svi = SVI(
            model=self.cut_posterior_net_model,
            guide=guide,
            optim=optimizer,
            loss=Trace_ELBO(),
        )

        self.rng_key, _ = random.split(self.rng_key)

        svi_results = svi.run(
            rng_key=self.rng_key,
            num_steps=self.n_iter_networks,
            progress_bar=self.progress_bar,
            data=self.data,
            init_params=init_vals,
        )

        map_params = guide.median(svi_results.params)

        self.rng_key, _ = random.split(self.rng_key)

        preds = Predictive(
            model=self.cut_posterior_net_model,
            guide=guide,
            params=svi_results.params,
            num_samples=1,
        )(self.rng_key, self.data)

        self.triu_star_probs = preds["triu_star_probs"][0]
        self.theta = map_params["theta"]
        self.gamma = map_params["gamma"]

    def init_triu_star_and_exposures(self):
        """
        Initialize triu_star and exposures using theta and gamma samples
        """

        self.rng_key, _ = random.split(self.rng_key)

        triu_star_samps = sample_posterior_triu_star(
            self.rng_key, self.triu_star_probs, self.n_nets_samples
        )

        # get exposures --> init is the posterior mean exposures
        exposures_samps = utils.vmap_compute_exposures(triu_star_samps, self.data.Z)

        self.exposures = exposures_samps.mean(axis=0)
        # self.exposures = jnp.median(exposures_samps, axis=0)

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
            num_chains=self.num_chains_outcome,
            # progress_bar=self.progress_bar,
            progress_bar=False,
        )

        self.rng_key, _ = random.split(self.rng_key)

        mcmc_plugin.run(
            self.rng_key, df_nodes, utils.Triu_to_mat(self.triu_star), self.data.Y
        )

        # Get posterior samples
        samples = mcmc_plugin.get_samples()

        self.eta = samples["eta"].mean(axis=0)
        self.rho = samples["rho"].mean()
        self.sig_inv = samples["sig_inv"].mean()

    def get_init_values(self):
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

        # print(
        #     "MWG init params:",
        #     "\n",
        #     "theta:",
        #     self.theta,
        #     "\n",
        #     "gamma:",
        #     self.gamma,
        #     "\n",
        #     #   "triu_star:", self.triu_star, "\n",
        #     "eta:",
        #     self.eta,
        #     "\n",
        #     "rho:",
        #     self.rho,
        #     "\n",
        #     "sig_inv:",
        #     self.sig_inv,
        # )

        if self.num_chains_outcome > 1:
            return replicate_params(init_params, self.num_chains_outcome)
        else:
            return init_params


# MWG sampler


class MWG_sampler:
    def __init__(
        self,
        rng_key,
        data,
        init_params,
        gwg_fn=make_gwg_gibbs_fn,
        combined_model=models.combined_model,
        n_warmup=3000,
        n_samples=2500,
        num_chains=4,
        continuous_sampler="NUTS",  # one of "NUTS" or "HMC"
        progress_bar=False,
    ):
        self.rng_key = random.split(rng_key)[0]
        self.data = data
        self.init_params = init_params
        self.combined_model = combined_model
        self.continuous_sampler = continuous_sampler
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.num_chains = num_chains
        self.progress_bar = progress_bar

        #  init function for mwg
        self.gwg_fn = gwg_fn(data)
        self.continuous_kernel = self.make_continuous_kernel()
        #  get posterior samples
        self.posterior_samples = self.get_samples()

        self.pred_func = Predictive(
            model=self.combined_model, posterior_samples=self.posterior_samples
        )

    def make_continuous_kernel(self):
        if self.continuous_sampler == "NUTS":
            return NUTS(self.combined_model)
        elif self.continuous_sampler == "HMC":
            return HMC(self.combined_model)
        else:
            raise ValueError("continuous_sampler must be one of 'NUTS' or 'HMC'")

    def get_samples(self):
        mwg_kernel = HMCGibbs(
            inner_kernel=self.continuous_kernel,
            gibbs_fn=self.gwg_fn,
            gibbs_sites=["triu_star"],
        )

        mwg_mcmc = MCMC(
            mwg_kernel,
            num_warmup=self.n_warmup,
            num_samples=self.n_samples,
            num_chains=self.num_chains,
            progress_bar=self.progress_bar,
        )

        self.rng_key, _ = random.split(self.rng_key)

        mwg_mcmc.run(self.rng_key, self.data, init_params=self.init_params)

        return mwg_mcmc.get_samples()

    def wasserstein_distance(self, true_vals):
        keys_to_keep = {"eta", "sig_inv", "rho", "triu_star"}
        # keys_to_keep = {"eta", "sig_inv", "triu_star"}
        post_samps = self.posterior_samples.copy()
        post_samps["rho"] = post_samps["rho"][:, None]
        post_samps["sig_inv"] = post_samps["sig_inv"][:, None]

        post_samps_k = {k: post_samps[k] for k in keys_to_keep}

        return utils.compute_1w_distance(post_samps_k, true_vals)

    def sample_pred_y(self, new_z):
        assert new_z.shape[0] == 2

        if new_z.ndim == 2:  # dynamic treatmets
            rng_keys = random.split(self.rng_key, 2)
            data_list = jax.vmap(utils.get_data_new_z, in_axes=(0, None))(
                new_z, self.data
            )
            pred_samples = jax.vmap(lambda rk, d: self.pred_func(rk, d)["Y"])(
                rng_keys, data_list
            )
            preds = pred_samples[0] - pred_samples[1]

        elif new_z.ndim == 3:  # stoch treatments
            n_approx = new_z.shape[1]
            rng_keys = random.split(self.rng_key, 2 * n_approx).reshape(2, n_approx, -1)
            vmap_func = jax.vmap(
                jax.vmap(
                    lambda zk, rk: self.pred_func(
                        rk, utils.get_data_new_z(zk, self.data)
                    )["Y"],
                    in_axes=(0, 0),  # Maps over n_approx
                ),
                in_axes=(0, 0),  # Maps over the intervention groups (2)
            )
            pred_samples = vmap_func(
                new_z, rng_keys
            )  # Expected shape (2, n_approx, m, n)
            preds_diff = pred_samples[0] - pred_samples[1]
            # return mean across n_approx (number of stoch treatments approx)
            preds = preds_diff.mean(axis=0)

        else:
            raise ValueError("new_z should have shape (2,n) or (2, n_approx, n)")

        return preds

    def new_intervention_error_stats(self, new_z, true_estimands, true_vals):
        wasser_dist = self.wasserstein_distance(true_vals)

        post_y_preds = self.sample_pred_y(new_z)
        return utils.compute_error_stats(post_y_preds, true_estimands, wasser_dist)

    # def new_intervention_error_stats(self, new_z, true_estimands, true_vals):
    #     wasser_dist = self.wasserstein_distance(true_vals)

    #     if new_z.ndim == 3:  # stoch intervention
    #         # compute exposures for new interventions
    #         expos_1 = utils.vmap_compute_exposures(
    #             self.posterior_samples["triu_star"], new_z[0, :, :]
    #         )
    #         expos_2 = utils.vmap_compute_exposures(
    #             self.posterior_samples["triu_star"], new_z[1, :, :]
    #         )
    #         expos_diff = expos_1 - expos_2
    #         # reshape expos_diff to have shape (n_stoch, M, N) where n_stoch is number of stoch treatments approx
    #         diff_shapes = expos_diff.shape
    #         expos_diff = expos_diff.reshape(
    #             diff_shapes[1], diff_shapes[0], diff_shapes[2]
    #         )
    #         z_diff = new_z[0, :, :] - new_z[1, :, :]
    #         estimates = utils.get_estimates_vmap(
    #             z_diff, expos_diff, self.posterior_samples["eta"]
    #         ).mean(axis=0)
    #         # estimates should have shape (M,n) where M is number of posterior samples
    #         return utils.compute_error_stats(estimates, true_estimands, wasser_dist)
    #     elif new_z.ndim == 2:  # dynamic intervention
    #         expos_1 = utils.vmap_compute_exposures(
    #             self.posterior_samples["triu_star"], new_z[0, :]
    #         )
    #         expos_2 = utils.vmap_compute_exposures(
    #             self.posterior_samples["triu_star"], new_z[1, :]
    #         )
    #         expos_diff = expos_1 - expos_2
    #         z_diff = new_z[0, :] - new_z[1, :]
    #         estimates = utils.get_estimates(
    #             z_diff, expos_diff, self.posterior_samples["eta"]
    #         )
    #         return utils.compute_error_stats(estimates, true_estimands, wasser_dist)
    #     else:
    #         raise ValueError("Invalid dimension for new interventions")
