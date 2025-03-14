import jax.numpy as jnp
import jax
from jax import random, vmap
from numpyro.infer import MCMC, NUTS, Predictive
import src.Models as models
import src.utils as utils

# --- MCMC for outcome model with fixed network (true or obs) ---


class mcmc_fixed_net:
    def __init__(
        self,
        rng_key,
        data,
        net_type,
        n_warmup=2000,
        n_samples=2500,
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

        self.df_nodes, self.adj_mat, self.triu = self.df_nodes_and_adj_mat()
        self.samples = self.get_samples()

        self.pred_func = Predictive(
            model=models.plugin_outcome_model, posterior_samples=self.samples
        )

    def df_nodes_and_adj_mat(self):
        if self.net_type == "true":
            df_nodes = jnp.transpose(
                jnp.stack(
                    [
                        jnp.ones(self.data.Z.shape[0]),
                        self.data.Z,
                        self.data.x,
                        self.data.true_exposures,
                    ]
                )
            )
            adj_mat = utils.Triu_to_mat(self.data.triu_star)
            triu = self.data.triu_star
        elif self.net_type == "obs":
            df_nodes = jnp.transpose(
                jnp.stack(
                    [
                        jnp.ones(self.data.Z.shape[0]),
                        self.data.Z,
                        self.data.x,
                        self.data.obs_exposures,
                    ]
                )
            )
            adj_mat = utils.Triu_to_mat(self.data.triu_obs)
            triu = self.data.triu_obs
        else:
            raise ValueError("net_type must be one of 'true' or 'obs'")
        return df_nodes, adj_mat, triu

    def get_samples(self):
        kernel_ = NUTS(models.plugin_outcome_model)
        mcmc_ = MCMC(
            kernel_,
            num_warmup=self.n_warmup,
            num_samples=self.n_samples,
            num_chains=self.num_chains,
            progress_bar=self.progress_bar,
        )
        mcmc_.run(self.rng_key, self.df_nodes, self.adj_mat, self.data.Y)

        return mcmc_.get_samples()

    def wasserstein_distance(self, true_vals):
        post_samps = self.samples.copy()
        post_samps["triu_star"] = jnp.repeat(
            jnp.array([self.triu]), self.n_samples * self.num_chains, axis=0
        )
        post_samps["rho"] = post_samps["rho"][:, None]
        post_samps["sig_inv"] = post_samps["sig_inv"][:, None]

        return utils.compute_1w_distance(post_samps, true_vals)

    def sample_pred_y(self, new_z):
        # TODO: make it happen the plugin outcome model with new z
        assert new_z.shape[0] == 2

        if new_z.ndim == 2:  # dynamic treatmets
            new_expos = utils.vmap_compute_exposures_new_z(self.triu, new_z)
            rng_keys = random.split(self.rng_key, 2)
            df_list = utils.vmap_df_new_z(new_z, new_expos, self.data.x)

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
                utils.vmap_compute_exposures_new_z, in_axes=(None, 0)
            )
            new_expos = vmap_exposures(self.triu, new_z)  # Shape (2, n_approx, n)

            vmap_df = jax.vmap(utils.vmap_df_new_z, in_axes=(0, 0, None))
            df_list = vmap_df(
                new_z, new_expos, self.data.x
            )  # Shape (2, n_approx, n, 4)

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

        else:
            raise ValueError("new_z should have shape (2, n) or (2, n_approx, n)")

        return preds

    def new_intervention_error_stats(self, new_z, true_estimands, true_vals):
        wasser_dist = self.wasserstein_distance(true_vals)

        post_y_preds = self.sample_pred_y(new_z)
        return utils.compute_error_stats(post_y_preds, true_estimands, wasser_dist)

    # def new_intervention_error_stats(self, new_z, true_estimands, true_vals):
    #     wasser_dist = self.wasserstein_distance(true_vals)

    #     if new_z.ndim == 3:  # stoch intervention
    #         # compute exposures for new interventions
    #         expos_1 = utils.compute_exposures(self.triu, new_z[0, :, :])
    #         expos_2 = utils.compute_exposures(self.triu, new_z[1, :, :])
    #         expos_diff = expos_1 - expos_2
    #         z_diff = new_z[0, :, :] - new_z[1, :, :]
    #         estimates = utils.get_estimates_vmap(
    #             z_diff, expos_diff, self.samples["eta"]
    #         ).mean(axis=0)
    #         # estimates should have shape (M,n) where M is number of posterior samples
    #         return utils.compute_error_stats(estimates, true_estimands, wasser_dist)
    #     elif new_z.ndim == 2:  # dynamic intervention
    #         expos_1 = utils.compute_exposures(self.triu, new_z[0, :])
    #         expos_2 = utils.compute_exposures(self.triu, new_z[1, :])
    #         expos_diff = expos_1 - expos_2
    #         z_diff = new_z[0, :] - new_z[1, :]
    #         estimates = utils.get_estimates(z_diff, expos_diff, self.samples["eta"])
    #         return utils.compute_error_stats(estimates, true_estimands, wasser_dist)
    #     else:
    #         raise ValueError("Invalid dimension for new interventions")
