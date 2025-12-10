import jax.numpy as jnp
from jax import random, jit
import jax

from numpyro.infer import MCMC, NUTS, HMC, Predictive
from numpyro.infer.hmc_gibbs import HMCGibbs


from src.GWG import IPState, IPInfo, propsal_logprobs
from src.MWG_sampler import sample_posterior_triu_star, replicate_params
import src.Models as models
import src.utils as utils

import Data.cs_aarhus.data_models as dm
import Data.cs_aarhus.util_data as ud


# Global hyper-parameters
BATCH_LEN = 1
N_STEPS = 1


# --- GWG sampler (LIP) ---


# Aux function for edge flip proposals
@jit
def weighted_sample_and_logprobs(key, scores):
    """
    Weighted sample with replacement of BATCH_LEN indices using Gumbel-max trick

    Args:
    key: PRNG key
    scores: Scores for the weights of the samples (log density ratios)

    """
    # Get samples using Gumbel-max trick
    gumbel_noise = random.gumbel(key, shape=scores.shape)
    perturbed = scores + gumbel_noise
    _, selected_indices = jax.lax.top_k(perturbed, BATCH_LEN)
    # Compute log soft-max probs of scores
    log_probs = jax.nn.log_softmax(scores)
    selected_log_probs = log_probs[selected_indices]
    # return indices and log probabilities
    return selected_indices, selected_log_probs.sum()


@jit
def GWG_step(rng_key, state, data, param):
    """
    One step of GWG sampler for A* parameter

    Args:
    rng_key: PRNG key
    state: IPState - current state of the sampler
    data: DataTuple - data for the model
    param: ParamTuple - parameters of the model

    Returns:
    IPState: new state of the sampler
    IPInfo: info about the acceptance of the proposal
    """
    key1, key2 = random.split(rng_key, 2)
    # sample idx to propose edge flip
    idx, forward_logprob = weighted_sample_and_logprobs(key1, state.scores)

    # proposed new state
    new_triu_star = state.positions.at[idx].set(1 - state.positions[idx])

    # backward proposal
    f_proposed, backward_grad = dm.triu_star_grad_fn(new_triu_star, data, param)
    backward_scores = (-(2 * new_triu_star - 1) * backward_grad) / 2
    backward_logprob = propsal_logprobs(idx, backward_scores)

    # accept/reject
    acceptance_ratio = jnp.minimum(
        jnp.exp(f_proposed - state.logdensity + backward_logprob - forward_logprob), 1
    )
    accept = random.uniform(key2) <= acceptance_ratio

    # update state and info
    new_triu_star = jax.lax.select(accept, new_triu_star, state.positions)
    new_logpost = jax.lax.select(accept, f_proposed, state.logdensity)
    new_grad = jax.lax.select(accept, backward_grad, state.logdensity_grad)
    new_scores = jax.lax.select(accept, backward_scores, state.scores)

    state = IPState(new_triu_star, new_logpost, new_grad, new_scores)
    info = IPInfo(acceptance_ratio, accept)

    return state, info


@jit
def GWG_kernel(rng_key, state, data, param):
    """
    Run N_STEPS of GWG sampler given the current param (continuous) values
    """

    def body_fun(carry, _):
        rng_key, cur_state = carry
        rng_key, step_key = random.split(rng_key)
        new_state, info = GWG_step(step_key, cur_state, data, param)
        return (rng_key, new_state), info

    # Run the scan
    (_, final_state), final_info = jax.lax.scan(
        body_fun, (rng_key, state), jnp.arange(N_STEPS)
    )

    return final_state, final_info


# Adapt the GWG sampler for the Metroplis-within-Gibbs (MWG) combined sampler (continuous and discrete parameters)


def make_gwg_gibbs_fn(data):
    """
    Will return a function that can be used as a Gibbs step for A* in the MWG sampler;
    The function will be used in the HMCGibbs sampler;
    It will depend on the current continuous parameters (gibbs_sites) and the fixed data
    """

    # Create a closure over the fixed data
    @jit  # Can JIT since data is fixed in closure
    def gwg_gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        # Get current state of triu_star
        cur_triu_star = jnp.astype(gibbs_sites["triu_star"], jnp.float32)

        # Get current values of continuous parameters
        cur_param = {
            "eta": hmc_sites["eta"],
            # "rho": hmc_sites["rho"],
            "sig_inv": hmc_sites["sig_inv"],
            "logits_star": hmc_sites["logits_star"],
        }

        # Create/update IPState
        cur_logdensity, cur_grad = dm.triu_star_grad_fn(cur_triu_star, data, cur_param)

        cur_scores = (-(2 * cur_triu_star - 1) * cur_grad) / 2

        state = IPState(
            positions=cur_triu_star,
            logdensity=cur_logdensity,
            logdensity_grad=cur_grad,
            scores=cur_scores,
        )

        # Run GWG (N_STEPS times)
        rng_key, _ = random.split(rng_key)
        new_state, _ = GWG_kernel(
            rng_key=rng_key, state=state, data=data, param=cur_param
        )

        # Return only the new positions as required by HMCGibbs
        # Note that we need to convert the positions to int32 as it is latent binary variable
        # See NumPyro conventions for coding latent variables
        return {"triu_star": jnp.astype(new_state.positions, jnp.int32)}

    return gwg_gibbs_fn


# --- MWG sampler ---


@jax.jit
def cut_post_log_density(triu_vals, post_probs):
    """
    Compute the log density of the posterior probabilities for the observed edges.

    Arguments:
    - triu_vals: n_layers x n choose 2 array of edge values.
    - post_probs: n choose 2 array of posterior probabilities.

    Returns:
    - log likelihood of the observed edges under the posterior probabilities.
    """
    return jnp.sum(
        triu_vals * jnp.log(post_probs) + (1 - triu_vals) * jnp.log1p(post_probs)
    )


vmap_post_density = jax.vmap(cut_post_log_density, in_axes=(0, None))


def max_post_net(triu_samps, probs):
    """
    Find the network with the highest posterior density.

    Arguments:
    - triu_samps: n_samples x n choose 2 array of posterior samples.
    - probs: n choose 2 array of posterior probabilities.

    Returns:
    - n choose 2 array of the network with the highest posterior density.
    """
    log_densities = vmap_post_density(triu_samps, probs)
    max_idx = jnp.argmax(log_densities)
    return triu_samps[max_idx]


class MWG_init:
    def __init__(
        self,
        rng_key,
        data,
        cut_posterior_net_model=dm.cutposterior_multilayer,
        cut_posterior_outcome_model=models.plugin_outcome_model,
        n_warmup=1000,
        n_samples=1000,
        num_chains=4,
        progress_bar=False,
        n_nets_samples=3000,
    ):
        self.rng_key = random.split(rng_key)[0]
        self.data = data
        self.cut_posterior_net_model = cut_posterior_net_model
        self.cut_posterior_outcome_model = cut_posterior_outcome_model

        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.num_chains = num_chains
        self.progress_bar = progress_bar

        self.n_nets_samples = n_nets_samples

        # initial parameters
        self.triu_star_probs = None
        self.triu_star = None
        self.exposures = None
        self.network_params = {}
        self.outcome_params = {}

    def init_network_params(self):
        """
        Initialize network parameters from cut-posterior
        """

        kernel = NUTS(self.cut_posterior_net_model)
        mcmc = MCMC(
            kernel,
            num_warmup=self.n_warmup,
            num_samples=self.n_samples,
            num_chains=self.num_chains,
            progress_bar=self.progress_bar,
        )

        self.rng_key, _ = random.split(self.rng_key)

        mcmc.run(self.rng_key, triu_vals=self.data["triu_vals"])

        samples = mcmc.get_samples()

        self.triu_star_probs = samples["probs_latent"].mean(axis=0)
        self.network_params = {
            k: v.mean(axis=0) for k, v in samples.items() if k != "probs_latent"
        }

    def init_triu_star_and_exposures(self):
        """
        Initialize triu_star and exposures using samples from network cut-posterior models
        """

        self.rng_key, _ = random.split(self.rng_key)

        triu_star_samps = sample_posterior_triu_star(
            self.rng_key, self.triu_star_probs, self.n_nets_samples
        )

        # get exposures --> init is the posterior mean exposures
        exposures_samps = ud.vmap_compute_exposures(triu_star_samps, self.data["Z"])

        self.exposures = exposures_samps.mean(axis=0)

        # get triu_star --> init is the posterior triu_star with largest log-posterior density
        self.triu_star = max_post_net(triu_star_samps, self.triu_star_probs)

    def init_outcome_model(self):
        """
        Initialize outcome model parameters from cut-posterior
        """
        df_nodes = ud.get_df_nodes(self.data["Z"], self.exposures)
        adj_mat = ud.triu_to_mat(self.triu_star)
        # adj_mat = ud.triu_to_mat(self.triu_star) + jnp.eye(
        #     ud.N_NODES
        # )  # add self-loops for stability

        # Define MCMC kernel and run MCMC
        kernel_plugin = NUTS(self.cut_posterior_outcome_model)
        mcmc_plugin = MCMC(
            kernel_plugin,
            num_warmup=self.n_warmup,
            num_samples=self.n_samples,
            num_chains=self.num_chains,
            progress_bar=self.progress_bar,
        )

        self.rng_key, _ = random.split(self.rng_key)

        mcmc_plugin.run(
            self.rng_key, df_nodes=df_nodes, adj_mat=adj_mat, Y=self.data["Y"]
        )

        # Get posterior samples
        samples = mcmc_plugin.get_samples()

        self.outcome_params = {k: v.mean(axis=0) for k, v in samples.items()}

    def get_init_values(self):
        """
        Get initial values for the MWG sampler


        Returns:
            dict with initial values for the MWG sampler
        """

        print("Initializing parameters for MWG sampler...")

        self.init_network_params()
        self.init_triu_star_and_exposures()
        self.init_outcome_model()

        init_params = (
            self.network_params | self.outcome_params | {"triu_star": self.triu_star}
        )

        print(
            "MWG init params:",
            "\n",
            "eta:",
            self.outcome_params["eta"],
            "\n",
            # "rho:",
            # self.outcome_params["rho"],
            # "\n",
            "sig_inv:",
            self.outcome_params["sig_inv"],
        )

        if self.num_chains > 1:
            return replicate_params(init_params, self.num_chains)
        else:
            return init_params


class MWG_sampler:
    def __init__(
        self,
        rng_key,
        data,
        init_params,
        gwg_fn=make_gwg_gibbs_fn,
        combined_model=dm.combined_model,
        n_warmup=1000,
        n_samples=1000,
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

        print("mean post eta:", self.posterior_samples["eta"].mean(axis=0))

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

        mwg_mcmc.run(self.rng_key, self.data, init_params=self.init_params)

        return mwg_mcmc.get_samples()

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
            data_list = jax.vmap(ud.get_data_new_z, in_axes=(0, None))(new_z, self.data)
            pred_samples = jax.vmap(lambda rk, d: self.pred_func(rk, d)["Y"])(
                rng_keys, data_list
            )
            preds = pred_samples[0] - pred_samples[1]

        elif new_z.ndim == 3:  # stoch treatments
            n_approx = new_z.shape[1]
            rng_keys = random.split(self.rng_key, 2 * n_approx).reshape(2, n_approx, -1)
            vmap_func = jax.vmap(
                jax.vmap(
                    lambda zk, rk: self.pred_func(rk, ud.get_data_new_z(zk, self.data))[
                        "Y"
                    ],
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


# --- mcmc with fixed network (true or not) ---


class mcmc_fixed_net:
    def __init__(
        self,
        rng_key,
        data,
        net_type,
        n_warmup=1000,
        n_samples=1000,
        num_chains=4,
        progress_bar=False,
    ):
        self.rng_key = rng_key
        self.data = data
        self.net_type = net_type
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.num_chains = num_chains
        self.progress_bar = progress_bar

        self.adj_mat = None
        self.triu = None
        self.df_nodes = None
        self.df_nodes_and_adj_mat()

        self.samples = self.get_samples()

        self.pred_func = Predictive(
            model=models.plugin_outcome_model, posterior_samples=self.samples
        )

    def df_nodes_and_adj_mat(self):
        if self.net_type == "true":
            self.adj_mat = ud.triu_to_mat(self.data["triu_star"])
            # self.adj_mat = ud.triu_to_mat(self.data["triu_star"]) + jnp.eye(
            #     ud.N_NODES
            # )  # add self-loops for stability
            self.triu = self.data["triu_star"]
        elif self.net_type == "agg_or":
            self.adj_mat = ud.triu_to_mat(self.data["agg_or_triu"])
            # self.adj_mat = ud.triu_to_mat(self.data["agg_or_triu"]) + jnp.eye(
            #     ud.N_NODES
            # )
            self.triu = self.data["agg_or_triu"]
        elif self.net_type == "agg_and":
            # self.adj_mat = ud.triu_to_mat(self.data["agg_and_triu"]) + jnp.eye(
            #     ud.N_NODES
            # )
            self.adj_mat = ud.triu_to_mat(self.data["agg_and_triu"])
            self.triu = self.data["agg_and_triu"]

        exposures = ud.compute_exposures(self.triu, self.data["Z"])
        self.df_nodes = ud.get_df_nodes(self.data["Z"], exposures)

    def get_samples(self):
        kernel_ = NUTS(models.plugin_outcome_model)
        mcmc_ = MCMC(
            kernel_,
            num_warmup=self.n_warmup,
            num_samples=self.n_samples,
            num_chains=self.num_chains,
            progress_bar=self.progress_bar,
        )
        mcmc_.run(self.rng_key, self.df_nodes, self.adj_mat, self.data["Y"])

        # mcmc_.print_summary()

        return mcmc_.get_samples()

    def wasserstein_distance(self, true_vals):
        post_samps = self.samples.copy()
        post_samps["triu_star"] = jnp.repeat(
            jnp.array([self.triu]), self.n_samples * self.num_chains, axis=0
        )
        # post_samps["rho"] = post_samps["rho"][:, None]
        post_samps["sig_inv"] = post_samps["sig_inv"][:, None]

        return utils.compute_1w_distance(post_samps, true_vals)

    def sample_pred_y(self, new_z):
        assert new_z.shape[0] == 2

        if new_z.ndim == 2:  # dynamic treatmets
            new_expos = ud.vmap_compute_exposures_new_z(self.triu, new_z)
            rng_keys = random.split(self.rng_key, 2)
            df_list = ud.vmap_df_nodes(new_z, new_expos)

            pred_samples = jax.vmap(
                lambda rk, df: self.pred_func(rk, df, self.adj_mat, None)["Y"]
            )(rng_keys, df_list)
            preds = pred_samples[0] - pred_samples[1]

        elif new_z.ndim == 3:
            # Case: new_z has shape (2, n_approx, n)
            n_approx = new_z.shape[1]

            # Generate independent RNG keys for all 2 * n_approx calls
            rng_keys = random.split(self.rng_key, 2 * n_approx).reshape(2, n_approx, -1)

            # Fully vectorized computation over (2, n_approx)
            vmap_exposures = jax.vmap(
                ud.vmap_compute_exposures_new_z, in_axes=(None, 0)
            )
            new_expos = vmap_exposures(self.triu, new_z)  # Shape (2, n_approx, n)

            vmap_df = jax.vmap(ud.vmap_df_nodes, in_axes=(0, 0))
            df_list = vmap_df(new_z, new_expos)  # Shape (2, n_approx, n, 4)

            # Fully vectorized function with correct in_axes
            vmap_pred = jax.vmap(
                jax.vmap(
                    lambda rk, df: self.pred_func(rk, df, self.adj_mat, None)["Y"],
                    in_axes=(0, 0),  # Maps over n_approx
                ),
                in_axes=(0, 0),  # Maps over intervention groups (2)
            )

            pred_samples = vmap_pred(rng_keys, df_list)  # Shape (2, n_approx, m, n)

            preds_diff = (
                pred_samples[0] - pred_samples[1]
            )  # Output shape (n_approx, m, n)
            preds = preds_diff.mean(axis=0)

        # if new_z.ndim == 3:  # stoch intervention
        #     # compute exposures for new interventions
        #     expos_1 = ud.compute_exposures(self.triu, new_z[0, :, :])
        #     expos_2 = ud.compute_exposures(self.triu, new_z[1, :, :])
        #     expos_diff = expos_1 - expos_2
        #     z_diff = new_z[0, :, :] - new_z[1, :, :]
        #     preds = ud.get_estimates_vmap(z_diff, expos_diff, self.samples["eta"]).mean(
        #         axis=0
        #     )
        # elif new_z.ndim == 2:  # dynamic intervention
        #     expos_1 = ud.compute_exposures(self.triu, new_z[0, :])
        #     expos_2 = ud.compute_exposures(self.triu, new_z[1, :])
        #     expos_diff = expos_1 - expos_2
        #     z_diff = new_z[0, :] - new_z[1, :]
        #     preds = ud.get_estimates(z_diff, expos_diff, self.samples["eta"])
        else:
            raise ValueError("new_z should have shape (2, n) or (2, n_approx, n)")

        return preds

    def new_intervention_error_stats(self, new_z, true_estimands, true_vals):
        wasser_dist = self.wasserstein_distance(true_vals)

        post_y_preds = self.sample_pred_y(new_z)
        return utils.compute_error_stats(post_y_preds, true_estimands, wasser_dist)
