###
# Metrpolis-within-Gibbs sampler with HMCGibbs
# GWG is used for discrete parameters (triu_star)
# Initalization of parameters from the cut-posterior
###

import jax
import jax.numpy as jnp
from jax import random, vmap
from numpyro.infer import (
    MCMC,
    NUTS,
    HMC,
    Predictive,
    SVI,
    Trace_ELBO,
)
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.optim import ClippedAdam
from numpyro.infer.hmc_gibbs import HMCGibbs
from numpyro.diagnostics import summary, print_summary
import numpyro.handlers as handlers
import time

import src.Models as models
import src.utils as utils
from src.GWG import (
    make_gwg_gibbs_fn,
    make_gwg_gibbs_fn_rep,
    GWG_kernel,
    IPState,
    ParamTuple,
)


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
        triu_star_grad_fn=models.triu_star_grad_fn,
        gwg_kernel_fn=GWG_kernel,
        n_iter_networks=20000,
        n_nets_samples=3000,
        n_warmup_outcome=2000,
        n_samples_outcome=3000,
        learning_rate=0.0005,
        num_chains_outcome=4,
        progress_bar=False,
        misspecified=False,
        refine_triu_star=True,
        gwg_init_steps=1000,
        gwg_init_batch_len=10,
    ):
        self.rng_key = random.split(rng_key)[0]
        self.data = data

        self.cut_posterior_net_model = cut_posterior_net_model
        self.cut_posterior_outcome_model = cut_posterior_outcome_model
        self.triu_star_log_posterior_fn = triu_star_log_posterior_fn
        self.triu_star_grad_fn = triu_star_grad_fn
        self.gwg_kernel_fn = gwg_kernel_fn

        self.n_iter_networks = n_iter_networks
        self.n_nets_samples = n_nets_samples
        self.n_warmup_outcome = n_warmup_outcome
        self.n_samples_outcome = n_samples_outcome
        self.learning_rate = learning_rate
        self.num_chains_outcome = num_chains_outcome
        self.progress_bar = progress_bar
        self.misspecified = misspecified

        self.refine_triu_star = refine_triu_star
        self.gwg_init_steps = gwg_init_steps
        self.gwg_init_batch_len = gwg_init_batch_len

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
        if self.misspecified:
            # Misspecified: Intercept, Treatment, Exposures (No X)
            df_nodes = jnp.transpose(
                jnp.stack(
                    [
                        jnp.ones(self.data.Y.shape[0]),
                        self.data.Z,
                        self.exposures,
                    ]
                )
            )
        else:
            # Correct Specification: Intercept, Treatment, X, Exposures
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
        # self.rho = samples["rho"].mean()
        self.sig_inv = samples["sig_inv"].mean()

    def refine_triu_star_values(self):
        """
        Refine the initial triu_star by running a short GWG chain
        conditioned on the initialized continuous parameters.
        """
        if self.progress_bar:
            print(f"Refining A* with {self.gwg_init_steps} GWG steps...")

        # 1. Package current continuous parameters
        cur_param = ParamTuple(
            theta=self.theta,
            gamma=self.gamma,
            eta=self.eta,
            # rho=self.rho,
            sig_inv=self.sig_inv,
        )

        # 2. Initialize IPState (gradients, scores)
        # We assume self.triu_star_grad_fn is set correctly (single vs repeated)
        cur_triu_star = jnp.astype(self.triu_star, jnp.float32)

        cur_logdensity, cur_grad = self.triu_star_grad_fn(
            cur_triu_star, self.data, cur_param
        )

        cur_scores = (-(2 * cur_triu_star - 1) * cur_grad) / 2

        state = IPState(
            positions=cur_triu_star,
            logdensity=cur_logdensity,
            logdensity_grad=cur_grad,
            scores=cur_scores,
        )

        # 3. Run GWG Kernel
        self.rng_key, _ = random.split(self.rng_key)

        start_t = time.time()

        new_state, _ = self.gwg_kernel_fn(
            rng_key=self.rng_key,
            state=state,
            data=self.data,
            param=cur_param,
            n_steps=self.gwg_init_steps,
            batch_len=self.gwg_init_batch_len,
        )

        new_state.positions.block_until_ready()
        end_t = time.time()

        new_triu_star_int = jnp.astype(new_state.positions, jnp.int32)

        # Compute Hamming distance (number of flipped edges)
        n_flips = jnp.sum(
            jnp.abs(new_triu_star_int - jnp.astype(self.triu_star, jnp.int32))
        )

        # Compute Log Density improvement
        # Note: new_state.logdensity is from the last step
        log_prob_diff = new_state.logdensity - cur_logdensity

        if self.progress_bar:
            print(f"  > Refinement done in {end_t - start_t:.4f}s")
            print(f"  > Edges flipped: {n_flips}")
            print(f"  > Log-prob change: {log_prob_diff:.4f}")

        # 4. Update self.triu_star
        self.triu_star = jnp.astype(new_state.positions, jnp.int32)

        # Optional: Update exposures based on new network
        # This keeps the object state consistent, though strictly optional for just init
        self.exposures = utils.compute_exposures(self.triu_star, self.data.Z)

    def get_init_values(self):
        """
        Get initial values for the MWG sampler.

        Args:
            compute_outcome_mcmc (bool):
                If True, runs MCMC to estimate 'eta', 'sig_inv'.
                If False, uses NumPyro's init_to_uniform for continuous params.
        """

        if self.progress_bar:
            print("Initializing parameters for MWG sampler...")

        self.init_network_params()
        self.init_triu_star_and_exposures()
        self.init_outcome_model()

        if self.refine_triu_star:
            self.refine_triu_star_values()

        init_params = {
            "theta": self.theta,
            "gamma": self.gamma,
            "triu_star": self.triu_star,
            "eta": self.eta,
            # "rho": self.rho,
            "sig_inv": self.sig_inv,
        }

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
        n_warmup=2000,
        n_samples=3000,
        num_chains=4,
        continuous_sampler="NUTS",  # one of "NUTS" or "HMC"
        progress_bar=False,
        misspecified=False,
        gwg_n_steps=1,
        gwg_batch_len=1,
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
        self.misspecified = misspecified

        self.sampling_time = None

        #  init function for mwg
        # self.gwg_fn = gwg_fn(data)
        self.gwg_fn = gwg_fn(data, n_steps=gwg_n_steps, batch_len=gwg_batch_len)
        self.continuous_kernel = self.make_continuous_kernel()
        #  get posterior samples
        self.posterior_samples = self.get_samples()

        self.pred_func = Predictive(
            model=self.combined_model, posterior_samples=self.posterior_samples
        )

    def make_continuous_kernel(self):
        if self.continuous_sampler == "NUTS":
            return NUTS(self.combined_model, target_accept_prob=0.9)
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

        # --- Timing Block ---
        start_time = time.time()
        # mwg_mcmc.run(self.rng_key, self.data, init_params=self.init_params)
        mwg_mcmc.run(
            self.rng_key,
            self.data,
            init_params=self.init_params,
        )

        samples = mwg_mcmc.get_samples()

        jax.tree_util.tree_leaves(samples)[
            0
        ].block_until_ready()  # Ensure all computations are done

        end_time = time.time()

        self.sampling_time = end_time - start_time

        return samples

        # return mwg_mcmc.get_samples()
        # return mwg_mcmc.get_samples(group_by_chain=True)

    def print_diagnostics(self, to_print=True):
        """
        Prints ESS, R-hat, and Computational Efficiency metrics.
        Excludes the high-dimensional 'triu_star' to prevent console overflow.
        """
        if to_print:
            print(f"\n=== MCMC Diagnostics ({self.continuous_sampler}) ===")

        # Filter out the large discrete network parameter
        cont_samples = {
            k: v for k, v in self.posterior_samples.items() if k != "triu_star"
        }

        # NumPyro summary (calculates ESS and R-hat)
        # We assume group_by_chain=False to get global stats across chains
        stats = summary(cont_samples, group_by_chain=False)
        if to_print:
            print_summary(cont_samples, group_by_chain=False)

        # --- Compute Efficiency Metric (ESS / Second) ---
        if to_print:
            print("\n--- Efficiency Metrics (ESS / sec) ---")

        min_ess_global = float("inf")
        mean_ess_eta = float("inf")

        for param_name, metrics in stats.items():
            # metrics['n_eff'] can be an array if the parameter is a vector (e.g. eta)
            n_eff = jnp.array(metrics["n_eff"])
            min_ess = jnp.min(n_eff)
            mean_ess = jnp.mean(n_eff)

            if param_name == "eta":
                mean_ess_eta = mean_ess

            # Update global minimum
            if min_ess < min_ess_global:
                min_ess_global = min_ess

            ess_per_sec = mean_ess / self.sampling_time
            if to_print:
                print(
                    f"{param_name:<10} | Mean ESS: {mean_ess:.1f} | Efficiency: {ess_per_sec:.2f} samples/sec"
                )
        min_ess_per_sec = min_ess_global / self.sampling_time
        if to_print:
            print("-" * 50)
            print(f"Global Min ESS/sec: {min_ess_per_sec:.4f}")
            print(f"Mean ESS for 'eta': {mean_ess_eta:.1f}")
        return min_ess_per_sec, mean_ess_eta, mean_ess_eta / self.sampling_time

    def wasserstein_distance(self, true_vals):
        # keys_to_keep = {"eta", "sig_inv", "rho", "triu_star"}
        keys_to_keep = {"eta", "sig_inv", "triu_star"}
        post_samps = self.posterior_samples.copy()
        # post_samps["rho"] = post_samps["rho"][:, None]
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
        if self.misspecified:
            wasser_dist = 999.0  # misspecified model, skip wasserstein distance
        else:
            wasser_dist = self.wasserstein_distance(true_vals)

        post_y_preds = self.sample_pred_y(new_z)
        return utils.compute_error_stats(post_y_preds, true_estimands, wasser_dist)
