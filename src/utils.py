###
# auxiliary functions
###

import jax.numpy as jnp
import jax
from jax import jit, vmap
import numpy as np
from typing import NamedTuple, Any
# from functools import partial


# --- Global variables ---

N = 500  # Number of nodes

# --- Data and Param namedtuples classes ---


class DataTuple(NamedTuple):
    x: jnp.ndarray
    x2: jnp.ndarray
    x_diff: jnp.ndarray
    x2_or: jnp.ndarray
    triu_star: jnp.ndarray
    triu_obs: jnp.ndarray
    triu_obs_rep: jnp.ndarray
    Z: jnp.ndarray
    obs_exposures: jnp.ndarray
    true_exposures: jnp.ndarray
    Y: Any


class ParamTuple(NamedTuple):
    theta: jnp.ndarray
    gamma: jnp.ndarray
    eta: jnp.ndarray
    # rho: float
    sig_inv: float


class NewEstimands(NamedTuple):
    Z_h: jnp.ndarray
    Z_stoch: jnp.ndarray
    Z_gate: jnp.ndarray
    estimand_h: jnp.ndarray
    estimand_stoch: jnp.ndarray
    estimand_gate: jnp.ndarray


# --- Aux functions for network wrangle and stats ---

# @jit
# def get_n_from_triu(triu_vals: jnp.ndarray) -> jnp.ndarray:
#     M = triu_vals.shape[0]
#     return jnp.int32((1 + jnp.sqrt(1 + 8 * M)) / 2)


@jit
def Triu_to_mat(triu_v):
    """
    Convert upper triangular vector to symmetric matrix
    """
    adj_mat = jnp.zeros((N, N))
    adj_mat = adj_mat.at[np.triu_indices(n=N, k=1)].set(triu_v)
    return adj_mat + adj_mat.T


@jit
def degree_centrality(adj_matrix):
    """
    Compute normalized degree centrality

    Parameters:
    adj_matrix (jnp.ndarray): Square adjacency matrix (n x n)

    Returns:
    jnp.ndarray: Vector of normalized degree centralities
    """
    # Compute degrees (sum of rows for undirected graph)
    degrees = jnp.sum(adj_matrix, axis=1)

    # Normalize by maximum possible degree (n-1)
    # n = adj_matrix.shape[0]
    # return degrees / (n - 1)
    return degrees / (N - 1)


@jit
def weighted_exposures(Z, node_weights, adj_mat):
    if Z.ndim == 1:  # Case when Z has shape (N,)
        return jnp.dot(adj_mat, Z * node_weights)
    elif Z.ndim == 2:  # Case when Z has shape (M, N)
        return jnp.dot(
            Z * node_weights, adj_mat.T
        )  # Transpose A_mat for correct dimensions


@jit
def compute_exposures(triu_star, Z):
    mat_star = Triu_to_mat(triu_star)
    deg_cen = degree_centrality(mat_star)
    return weighted_exposures(Z, deg_cen, mat_star)
    # deg_cen = degree_centrality(mat_star)
    # return weighted_exposures(Z, jnp.ones(N), mat_star)


vmap_compute_exposures = vmap(compute_exposures, in_axes=(0, None))

vmap_compute_exposures_new_z = vmap(compute_exposures, in_axes=(None, 0))


def get_data_new_z(new_z, data):
    """
    Get new data tuple for new interventions

    Args:
    new_z: new interventions with shape (n,)
    data: data tuple

    """
    return DataTuple(
        x=data.x,
        x2=data.x2,
        x_diff=data.x_diff,
        x2_or=data.x2_or,
        triu_star=data.triu_star,
        triu_obs=data.triu_obs,
        triu_obs_rep=data.triu_obs_rep,
        Z=new_z,
        obs_exposures=data.obs_exposures,
        true_exposures=data.true_exposures,
        Y=None,
    )


@jit
def df_node_new_z(new_z, new_expos, x):
    n = x.shape[0]
    return jnp.transpose(
        jnp.stack(
            [
                jnp.ones(n),
                new_z,
                x,
                new_expos,
            ]
        )
    )


vmap_df_new_z = vmap(df_node_new_z, in_axes=(0, 0, None))


@jit
def get_estimates(z_diff, expos_diff, eta_samps):
    """
    Get estimates of new interventions given posterior 'eta' samples
    Args:
    new_z: New treatment assignments jnp.ndarray with shape (2, n)
    expos_diff: Difference in exposures jnp.ndarray with shape of (M, n) or (n,) (multi vs fixed nets)
    eta_samps: Samples of posterior eta parameters with shape (M, 4) where M is number of posterior draws

    Returns:
    jnp.ndarray: Estimated expected outcomes; shape (M, N) where M is the number of samples
    """
    if expos_diff.ndim == 2:
        return (
            eta_samps[:, 1][:, None] * z_diff[None, :]
            + eta_samps[:, 3][:, None] * expos_diff
        )
    elif expos_diff.ndim == 1:
        return (
            eta_samps[:, 1][:, None] * z_diff[None, :]
            + eta_samps[:, 3][:, None] * expos_diff[None, :]
        )


# estimates for stochastic interventions with multiple treatments and expos diff
get_estimates_vmap = vmap(get_estimates, in_axes=(0, 0, None))


def compute_error_stats(post_estimates, true_estimand, wasserstein_dist):
    """
    Compute error metrics for posterior estimates

    Args:
    post_estimates: Posterior estimates with shape (M, N) where M is the number of samples
    true_estimand: True estimand with shape (N,)
    wasserstein_dist: Wasserstein distance between posterior samples and true values

    """
    # mean values
    mean_estimand = jnp.nanmean(true_estimand)  # scalar
    mean_of_units = jnp.nanmean(post_estimates, axis=1)  # shape (M,)
    mean_of_samples = jnp.nanmean(post_estimates, axis=0)  # shape (N,)
    mean_all = jnp.nanmean(post_estimates)  # scalar
    median_all = jnp.nanmedian(post_estimates)  # scalar

    # raw errors
    units_error = mean_of_units - mean_estimand
    units_rel_error = (mean_of_units - mean_estimand) / mean_estimand
    unit_level_error = mean_of_samples - true_estimand
    unit_level_rel_error = (mean_of_samples - true_estimand) / (true_estimand + 1e-6)

    # error metrics
    rmse = jnp.round(jnp.sqrt(jnp.nanmean(jnp.square(units_error))), 5)
    rmse_rel = jnp.round(jnp.sqrt(jnp.nanmean(jnp.square(units_rel_error))), 5)
    # mae = jnp.round(jnp.mean(jnp.abs(units_error)), 5)
    mae = jnp.round(jnp.nanmean(jnp.abs(unit_level_error)), 5)
    # mape = jnp.round(jnp.mean(jnp.abs(units_rel_error)), 5)
    mape = jnp.round(jnp.nanmean(jnp.abs(unit_level_rel_error)), 5)

    bias = jnp.round(mean_all - mean_estimand, 5)
    std = jnp.round(jnp.nanstd(mean_of_units), 5)

    # coverage
    q025 = jnp.nanquantile(mean_of_units, 0.025)
    q975 = jnp.nanquantile(mean_of_units, 0.975)
    coverage = (q025 <= mean_estimand) & (mean_estimand <= q975)

    q025_ind = jnp.nanquantile(post_estimates, 0.025, axis=0)
    q975_ind = jnp.nanquantile(post_estimates, 0.975, axis=0)
    coverage_ind = (q025_ind <= true_estimand) & (true_estimand <= q975_ind)
    mean_cover_ind = jnp.round(jnp.nanmean(coverage_ind), 5)

    return {
        "mean": jnp.round(mean_all, 5),
        "median": jnp.round(median_all, 5),
        "true": jnp.round(mean_estimand, 5),
        "bias": bias,
        "std": std,
        "RMSE": rmse,
        "RMSE_rel": rmse_rel,
        "MAE": mae,
        "MAPE": mape,
        "q025": q025,
        "q975": q975,
        "covering": coverage,
        "mean_ind_cover": mean_cover_ind,
        "w_dist": wasserstein_dist,
    }


@jax.jit
def compute_1w_distance(
    posterior_samples: dict, true_vals: dict, lambda_discrete: float = 1.0
) -> float:
    """
    Compute the 1-Wasserstein distance (average distance) between
    posterior samples and the true parameters, assuming exactly one
    discrete key called 'triu_star' and all other keys are continuous.

    Parameters
    ----------
    posterior_samples : dict of jnp.ndarray
        Each entry has shape (M, k_i).
        E.g.:
          - 'theta': (M, 2)
          - 'eta':   (M, 4)
          - ...
          - 'triu_star': (M, 124750)  <-- discrete/binary or categorical

    true_vals : dict of jnp.ndarray
        Each entry has shape (k_i,).
        E.g.:
          - 'theta': (2,)
          - 'eta':   (4,)
          - ...
          - 'triu_star': (124750,)

    lambda_discrete : float
        Weight for discrete mismatch distance.

    Returns
    -------
    float
        The 1-Wasserstein distance = mean of the mixed distance
        from each sample to the true parameters (point mass).
    """
    # Get all keys from the posterior_samples dict
    sample_keys = list(posterior_samples.keys())
    # Assume M (number of samples) is consistent across keys
    M = posterior_samples[sample_keys[0]].shape[0]

    # This function computes the distance for the sample at index `m`
    def single_sample_distances(m: int) -> float:
        dist_m = 0.0
        # Loop over every key in the dict
        for key in sample_keys:
            sample_val = posterior_samples[key][m]  # shape (k_i,)
            true_val = true_vals[key]  # shape (k_i,)

            if key == "triu_star":
                # Discrete distance:
                # Fraction of mismatches (Hamming distance normalized by length)
                mismatches = jnp.sum(sample_val != true_val)
                frac_mismatch = mismatches / sample_val.size
                dist_m += lambda_discrete * frac_mismatch
            else:
                # Continuous distance (Euclidean norm)
                dist_m += jnp.linalg.norm(sample_val - true_val, ord=2)

        return dist_m

    # Vectorize the distance computation over all sample indices [0..M-1]
    indices = jnp.arange(M)
    distances = jax.vmap(single_sample_distances)(indices)

    # 1-Wasserstein distance = mean of these distances
    return jnp.mean(distances)
