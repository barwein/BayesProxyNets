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
    n = adj_matrix.shape[0]
    return degrees / (n - 1)


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
    return compute_exposures(Z, deg_cen, mat_star)


vmap_compute_exposures = vmap(compute_exposures, in_axes=(0, None))


@jit
def get_estimates(new_z, triu_star_samps, eta_samps):
    """
    Get estimates of the outcome model given new treatment assignments

    Args:
    new_z: New treatment assignments
    triu_star_samps: Samples of upper triangular adjacency matrix
    eta_samps: Samples of eta parameters

    Returns:
    jnp.ndarray: Estimated expected outcomes; shape (M, N) where M is the number of samples
    """
    exposures_samps = vmap_compute_exposures(triu_star_samps, new_z)
    estimates = (
        eta_samps[:, 1][:, None] * new_z + eta_samps[:, 3][:, None] * exposures_samps
    )
    return estimates


# TODO: def compute_error_stats(....):
