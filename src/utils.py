###
# auxiliary functions
###

import jax.numpy as jnp
import jax
from jax import jit, vmap
import numpy as np
from typing import NamedTuple
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
    # TODO: triu_obs_rep: jnp.ndarray
    Z: jnp.ndarray
    obs_exposures: jnp.ndarray
    true_exposures: jnp.ndarray
    Y: jnp.ndarray


class ParamTuple(NamedTuple):
    theta: jnp.ndarray
    gamma: jnp.ndarray
    eta: jnp.ndarray
    rho: float
    sig_inv: float


class NewEstimands(NamedTuple):
    Z_h: jnp.ndarray
    Z_stoch: jnp.ndarray
    estimand_h: jnp.ndarray
    estimand_stoch: jnp.ndarray


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


vmap_compute_exposures = vmap(compute_exposures, in_axes=(0, None))


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
        return eta_samps[:, 1][:, None] * z_diff[None, :] + eta_samps[:, 3][:, None] * expos_diff
    elif expos_diff.ndim == 1:
        return eta_samps[:, 1][:, None] * z_diff[None, :] + eta_samps[:, 3][:, None] * expos_diff[None, :]

# estimates for stochastic interventions with multiple treatments and expos diff
get_estimates_vmap = vmap(get_estimates, in_axes=(0, 0, None))


def compute_error_stats(post_estimates, true_estimand):
    """
    Compute error metrics for posterior estimates

    Args:
    post_estimates: Posterior estimates with shape (M, N) where M is the number of samples
    true_estimand: True estimand with shape (N,)

    """
    # mean values
    mean_estimand = jnp.mean(true_estimand)  # scalar
    mean_of_units = jnp.mean(post_estimates, axis=1)  # shape (M,)
    mean_all = jnp.mean(post_estimates)  # scalar
    median_all = jnp.median(post_estimates)  # scalar

    # raw errors
    units_error = mean_of_units - mean_estimand
    units_rel_error = (mean_of_units - mean_estimand) / mean_estimand

    # error metrics
    rmse = jnp.round(jnp.sqrt(jnp.mean(jnp.square(units_error))), 5)
    rmse_rel = jnp.round(jnp.sqrt(jnp.mean(jnp.square(units_rel_error))), 5)
    mae = jnp.round(jnp.mean(jnp.abs(units_error)), 5)
    mape = jnp.round(jnp.mean(jnp.abs(units_rel_error)), 5)

    bias = jnp.round(mean_all - mean_estimand, 5)
    std = jnp.round(jnp.std(mean_of_units), 5)

    # coverage
    q025 = jnp.quantile(mean_of_units, 0.025)
    q975 = jnp.quantile(mean_of_units, 0.975)
    coverage = (q025 <= mean_estimand) & (mean_estimand <= q975)

    q025_ind = jnp.quantile(post_estimates, 0.025, axis=0)
    q975_ind = jnp.quantile(post_estimates, 0.975, axis=0)
    coverage_ind = (q025_ind <= true_estimand) & (true_estimand <= q975_ind)
    mean_cover_ind = jnp.round(jnp.mean(coverage_ind), 5)

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
    }
