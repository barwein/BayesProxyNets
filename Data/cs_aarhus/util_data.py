import jax.numpy as jnp
import jax
from jax import random
import numpy as np
import Simulations.data_gen as dg
import src.utils as utils


# General parameters

N_NODES = 61

# --- Read data ---


def read_layers_data():
    layers = {}
    with open("Data/cs_aarhus/CS-Aarhus_layers.txt", "r") as f:
        # Skip header
        next(f)
        for line in f:
            layer_id_str, layer_label = line.strip().split()
            layer_id = int(layer_id_str)
            layers[layer_id] = layer_label
    return layers


def read_nodes_data():
    idx_to_node = []
    with open("Data/cs_aarhus/CS-Aarhus_nodes.txt", "r") as f:
        next(f)  # skip header
        for line in f:
            node_id_str, node_label = line.strip().split()
            node_id = int(node_id_str)
            idx_to_node.append(node_id)

    node_to_idx = {node_id: i for i, node_id in enumerate(idx_to_node)}

    return node_to_idx


def read_edgelist_data():
    adjacency_matrices = [
        jnp.zeros((N_NODES, N_NODES), dtype=jnp.float32) for _ in range(5)
    ]
    node_to_idx = read_nodes_data()
    with open("Data/cs_aarhus/CS-Aarhus_multiplex.edges", "r") as f:
        for line in f:
            # Each line looks like: layerID node1 node2 weight
            parts = line.strip().split()
            lay = int(parts[0])
            n1 = int(parts[1])
            n2 = int(parts[2])
            w = float(parts[3])  # always 1 according to the description

            # Convert node IDs from 1-based to 0-based for indexing
            row = node_to_idx[n1]
            col = node_to_idx[n2]
            layer_idx = lay - 1  # also 0-based

            adjacency_matrices[layer_idx] = (
                adjacency_matrices[layer_idx].at[row, col].set(w)
            )
            adjacency_matrices[layer_idx] = (
                adjacency_matrices[layer_idx].at[col, row].set(w)
            )

    return jnp.array(adjacency_matrices)


def get_adj_mat_dict():
    layers = read_layers_data()
    adjacency_matrices = read_edgelist_data()
    adj_mat_dict = {layers[i + 1]: adjacency_matrices[i] for i in range(5)}
    # removed 'coauthor' layer as it is contained in the other layers
    adj_mat_dict_filtered = {k: v for k, v in adj_mat_dict.items() if k != "coauthor"}
    return adj_mat_dict_filtered


def get_triu_dict(networks_dicts):
    triu_indices = jnp.triu_indices(N_NODES, k=1)  # Indices for upper triangle
    triu_dict = {name: adj[triu_indices] for name, adj in networks_dicts.items()}
    # triu_array = jnp.stack(list(triu_dict.values()))
    return triu_dict


def network_data():
    adj_mat_dict = get_adj_mat_dict()
    triu_dict = get_triu_dict(adj_mat_dict)
    return {
        "adj_mat_dict": adj_mat_dict,
        "triu_dict": triu_dict,
    }


def triu_array_obs_n_latent(triu_dict: dict, latent_layer: str) -> tuple:
    """Splits triu_dict into observed layers and latent layer.

    Args:
        triu_dict (dict): Dictionary with keys as network names and values as 1D JAX arrays of upper triangular elements.
        latent_layer (str): The key corresponding to the latent network.

    Returns:
        tuple:
            - observed_array (jnp.ndarray) of shape (3, m), containing all networks except the latent layer.
            - latent_array (jnp.ndarray) of shape (m,), containing the latent network's upper triangular values.
    """
    latent_array = triu_dict[latent_layer]  # Shape (m,)

    # Extract observed triu (excluding latent_layer)
    observed_values = [v for k, v in triu_dict.items() if k != latent_layer]

    observed_array = jnp.array(observed_values)  # Shape (3, m)

    return observed_array, latent_array


# --- util functions ---


@jax.jit
def triu_to_mat(triu):
    mat = jnp.zeros((N_NODES, N_NODES))
    mat = mat.at[jnp.triu_indices(n=N_NODES, k=1)].set(triu)
    return mat + mat.T


@jax.jit
def compute_deg(triu):
    mat = triu_to_mat(triu)
    return jnp.sum(mat, axis=1)


vmap_deg = jax.vmap(compute_deg)


def get_df_nodes(Z, expos):
    return jnp.transpose(jnp.stack([jnp.ones(N_NODES), Z, expos]))


vmap_df_nodes = jax.vmap(get_df_nodes, in_axes=(0, 0))


def get_data_new_z(new_z, data):
    """
    Get new data dict for new interventions
    'Y' is None for sampling from its predictive distribution

    Args:
    new_z: new interventions with shape (n,)
    data: data tuple

    """
    return {
        "Z": new_z,
        "Y": None,
        "triu_vals": data["triu_vals"],
    }


# --- Aggregate multilayers networks (OR/AND) ---


def aggregate_edges(triu_vals, method="or"):
    """
    Aggregate edges from multiple layers into a single adjacency matrix (upper triangle values)

    Arguments:
    - triu_vals: n_layers x n choose 2 array of edge values.
    - method: "or" or "and" for how to aggregate edges.

    Returns:
    -upper triangle (triu) values of adjacency matrix of aggregated edges.
    """
    if method == "or":
        return jnp.any(triu_vals, axis=0)
    elif method == "and":
        return jnp.all(triu_vals, axis=0)
    else:
        raise ValueError("Invalid aggregation method. Must be 'or' or 'and'.")


# --- Generate treatments and outcome data ---


@jax.jit
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

    # return degrees / (n - 1)
    return degrees / (N_NODES - 1)


@jax.jit
def compute_exposures(triu_star, Z):
    """
    Compute weighted exposures for each node

    Parameters:
    triu_star (jnp.ndarray): Upper triangular adjacency matrix of latent network
    Z (jnp.ndarray): Treatment assignments with shape (n,)

    Returns:
    jnp.ndarray: Vector of weighted exposures for each node

    """
    mat_star = triu_to_mat(triu_star)
    deg_cen = degree_centrality(mat_star)
    return utils.weighted_exposures(Z, deg_cen, mat_star)


vmap_compute_exposures = jax.vmap(compute_exposures, in_axes=(0, None))

vmap_compute_exposures_new_z = jax.vmap(compute_exposures, in_axes=(None, 0))


def CAR_cov(triu_vals, sig_inv, rho):
    """
    Compute the covariance matrix of outcomes Y in a Conditional Auto-regressive (CAR) model
    """
    # Cov(Y) = \Sigma = sig_inv * (D - rho*A)^-1
    # So precision = \Sigma^{-1} = (1/sig_inv) * (D - rho*A)
    adj_mat = triu_to_mat(triu_vals)
    adj_mat += jnp.eye(N_NODES)  # Add self-loops for stability
    degs_diag = jnp.sum(adj_mat, axis=1) * jnp.eye(N_NODES)
    # Compute precision matrix Sigma^{-1}
    precis_ = sig_inv * (degs_diag - rho * adj_mat)
    # Return Sigma
    return jnp.linalg.inv(precis_)


def generate_data(key, triu_star, eta, rho, sig_inv):
    """
    Generate synthetic data for treatments and outcomes
    """
    z_key, y_key = random.split(key)
    Z = dg.generate_treatments(rng=z_key, n=N_NODES)
    expos = compute_exposures(triu_star, Z)
    df_nodes = get_df_nodes(Z, expos)
    mean_y = df_nodes @ eta
    y_cov = CAR_cov(triu_star, sig_inv, rho)
    # y_cov = random.normal(y_key, shape=(n,))*sig_inv
    # print("y_cov is positive definite?", jnp.all(jnp.linalg.eigvals(y_cov) > 0))
    assert jnp.all(jnp.linalg.eigvals(y_cov) > 0), (
        "Covariance matrix is not positive definite"
    )

    y = random.multivariate_normal(y_key, mean_y, y_cov)
    # y = mean_y + y_cov

    return {
        "Z": Z,
        "true_exposures": expos,
        "Y": y,
    }


def get_true_estimands(n, z_new, triu_star, eta):
    """
    compute true estimand values given new interventions
    """
    if z_new.ndim == 3:  # stoch intervention
        exposures_new1 = compute_exposures(triu_star, z_new[0, :, :])
        exposures_new2 = compute_exposures(triu_star, z_new[1, :, :])
        exposures_diff = exposures_new1 - exposures_new2
        z_diff = z_new[0, :, :] - z_new[1, :, :]
        n_stoch = z_new.shape[1]
        results = np.zeros((n_stoch, n))
        for i in range(n_stoch):
            results[i, :] = eta[1] * z_diff[i, :] + eta[2] * exposures_diff[i, :]
        return jnp.mean(results, axis=0).squeeze()
    elif z_new.ndim == 2:  # dynamic intervention
        exposures_new1 = compute_exposures(triu_star, z_new[0, :])
        exposures_new2 = compute_exposures(triu_star, z_new[1, :])
        exposures_diff = exposures_new1 - exposures_new2
        z_diff = z_new[0, :] - z_new[1, :]
        results = eta[1] * z_diff + eta[2] * exposures_diff
        return results
    else:
        raise ValueError("Invalid dimension for new interventions")


def get_intervention_estimand(key, triu_star, eta, n_approx=100):
    # new stochastic intervention
    Z_stoch = dg.stochastic_intervention(key, n=N_NODES, n_approx=n_approx)

    # new estimand
    stoch_estimands = get_true_estimands(N_NODES, Z_stoch, triu_star, eta)

    return {
        "Z_stoch": Z_stoch,
        "estimand_stoch": stoch_estimands,
    }


@jax.jit
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
            + eta_samps[:, 2][:, None] * expos_diff
        )
    elif expos_diff.ndim == 1:
        return (
            eta_samps[:, 1][:, None] * z_diff[None, :]
            + eta_samps[:, 2][:, None] * expos_diff[None, :]
        )


# estimates for stochastic interventions with multiple treatments and expos diff
get_estimates_vmap = jax.vmap(get_estimates, in_axes=(0, 0, None))
