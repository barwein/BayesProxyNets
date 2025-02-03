import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
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

    def new_intervention_error_stats(self, new_z, true_estimands):
        if new_z.ndim == 3:  # stoch intervention
            # compute exposures for new interventions
            expos_1 = utils.compute_exposures(self.triu, new_z[0, :, :])
            expos_2 = utils.compute_exposures(self.triu, new_z[1, :, :])
            expos_diff = expos_1 - expos_2
            z_diff = new_z[0, :, :] - new_z[1, :, :]
            estimates = utils.get_estimates_vmap(
                z_diff, expos_diff, self.samples["eta"]
            ).mean(axis=0)
            # estimates should have shape (M,n) where M is number of posterior samples
            return utils.compute_error_stats(estimates, true_estimands)
        elif new_z.ndim == 2:  # dynamic intervention
            expos_1 = utils.compute_exposures(self.triu, new_z[0, :])
            expos_2 = utils.compute_exposures(self.triu, new_z[1, :])
            expos_diff = expos_1 - expos_2
            z_diff = new_z[0, :] - new_z[1, :]
            estimates = utils.get_estimates(z_diff, expos_diff, self.samples["eta"])
            return utils.compute_error_stats(estimates, true_estimands)
        else:
            raise ValueError("Invalid dimension for new interventions")
