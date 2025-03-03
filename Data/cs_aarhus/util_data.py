import jax.numpy as jnp
import jax
from jax import random
import networkx as nx
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


# --- Aggregate multilayers networks (OR/AND) ---


def aggregate_edges(triu_vals, method="or"):
    """
    Aggregate edges from multiple layers into a single adjacency matrix.

    Arguments:
    - triu_vals: n_layers x n choose 2 array of edge values.
    - method: "or" or "and" for how to aggregate edges.

    Returns:
    - n x n adjacency matrix of aggregated edges.
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
    mat_star = triu_to_mat(triu_star)
    deg_cen = degree_centrality(mat_star)
    return utils.weighted_exposures(Z, deg_cen, mat_star)


vmap_compute_exposures = jax.vmap(compute_exposures, in_axes=(0, None))


def CAR_cov(triu_vals, sig_inv, rho):
    # Cov(Y) = \Sigma = sig_inv * (D - rho*A)^-1
    # So precision = \Sigma^{-1} = (1/sig_inv) * (D - rho*A)
    adj_mat = triu_to_mat(triu_vals)
    adj_mat += jnp.eye(N_NODES)  # Add self-loops for stability
    degs_diag = jnp.sum(adj_mat, axis=1) * jnp.eye(N_NODES)
    # Compute precision matrix Sigma^{-1}
    precis_ = sig_inv * (degs_diag - rho * adj_mat)
    # Return Sigma
    return jnp.linalg.inv(precis_)


def get_df_nodes(Z, expos):
    return jnp.transpose(jnp.stack([jnp.ones(N_NODES), Z, expos]))


def generate_data(key, triu_star, eta, rho, sig_inv):
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
