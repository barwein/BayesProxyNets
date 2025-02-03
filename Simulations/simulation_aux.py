# TODO: write aux function that will wrap the simulation study

# TODO: def one_iter_fixed_network


# TODO: def one_iter_mwg

# TODO: make all functions work with repeated measures of triu_obs


import jax.numpy as jnp
import jax
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import src.Models as models
import src.utils as utils

# --- MCMC for outcome model with fixed network (true or obs) ---


def one_iter_fixed_network(
    rng_key, data, net_type, n_warmup=2000, n_samples=2500, num_chains=4
):
    """
    Run MCMC for outcome model with fixed network (true or obs)

    Args:
    rng_key: JAX PRNG key
    data: DataTuple object
    net_type: str, one of "true" or "obs"
    n_warmup: number of warmup iterations
    n_samples: number of samples
    num_chains: number of chains

    Returns:
    dict with posterior samples

    """
    # df nodes and adj_mat
    if net_type == "true":
        df_nodes = jnp.transpose(
            jnp.stack([jnp.ones(data.Z.shape[0]), data.Z, data.x, data.true_exposures])
        )
        adj_mat = utils.Triu_to_mat(data.triu_star)
    elif net_type == "obs":
        df_nodes = jnp.transpose(
            jnp.stack([jnp.ones(data.Z.shape[0]), data.Z, data.x, data.obs_exposures])
        )
        adj_mat = utils.Triu_to_mat(data.triu_obs)
    else:
        raise ValueError("net_type must be one of 'true' or 'obs'")

    # Define MCMC kernel and run MCMC
    kernel_ = NUTS(models.plugin_outcome_model)
    mcmc_ = MCMC(
        kernel_,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=num_chains,
        progress_bar=False,
    )

    mcmc_.run(rng_key, df_nodes, adj_mat, data.Y)

    # Get posterior samples
    samples = mcmc_.get_samples()

    return samples


def outcome_mcmc_fixed_net(
    rng_key, data, net_type, n_warmup=2000, n_samples=2500, num_chains=4
):
    """
    Run MCMC for outcome model with fixed network (true or obs)

    Args:
    rng_key: JAX PRNG key
    data: DataTuple object
    net_type: str, one of "true" or "obs"
    n_warmup: number of warmup iterations
    n_samples: number of samples
    num_chains: number of chains

    Returns:
    dict with posterior samples

    """
    post_samples = one_iter_fixed_network(
        rng_key, data, net_type, n_warmup, n_samples, num_chains
    )

    