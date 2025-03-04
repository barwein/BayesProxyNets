import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import jax
import numpyro
import numpyro.distributions as dist

import Data.cs_aarhus.util_data as ud
import src.Models as models


# Global variables
N_NODES = 61
PRIOR_SCALE = 3.0

# --- Cut-posterior models ---


def cutposterior_multilayer(triu_vals, K=2):
    """
    Cut-posterior multilayer networks model
    p(\cA| \theta, V)p(A* | \theta, V)

    Args:
    triu_vals: n_layers x (n choose 2) array of edge values
    K: Dimension of the latent space
    """
    n_layers = triu_vals.shape[0]

    # ------------------------
    # Hyper-priors (hierarchical)
    # ------------------------
    mu0 = numpyro.sample("mu0", dist.Normal(0.0, PRIOR_SCALE))
    mu1 = numpyro.sample("mu1", dist.Normal(0.0, PRIOR_SCALE))

    sigma0 = numpyro.sample("sigma0", dist.Gamma(2.0, 2.0))  # scale for theta0
    sigma1 = numpyro.sample("sigma1", dist.Gamma(2.0, 2.0))  # scale for theta1

    # We have n_layers+1 sets of (theta0_k, theta1_k):
    #   k=0,...,n_layers-1 for the observed networks
    #   k=n_layers for the *latent* (unobserved) network
    theta0 = []
    theta1 = []
    for k in range(n_layers + 1):
        theta0_k_nrm = numpyro.sample(f"theta0_{k}_nrm", dist.Normal())
        # theta0_k = numpyro.deterministic(f"theta0_{k}", theta0_k_nrm * sigma0 + mu0)
        theta0_k = theta0_k_nrm * sigma0 + mu0
        # t1_k = numpyro.sample(f"theta1_{k}", dist.Normal(mu1, sigma1))
        theta1_k_nrm = numpyro.sample(f"theta1_{k}_nrm", dist.Normal())
        # theta1_k = numpyro.deterministic(f"theta1_{k}", theta1_k_nrm * sigma1 + mu1)
        theta1_k = theta1_k_nrm * sigma1 + mu1
        # t1_k = t1_k * sigma1 + mu1
        theta0.append(theta0_k)
        theta1.append(theta1_k)

    # # ------------------------
    # # Shared latent positions
    # # ------------------------
    # shape: (n, K)
    with numpyro.plate("nodes", N_NODES):
        V = numpyro.sample("V", dist.MultivariateNormal(0.0, jnp.eye(K)))

    idx = jnp.triu_indices(n=N_NODES, k=1)
    V_diff = V[idx[0]] - V[idx[1]]
    V_norm = jnp.linalg.norm(V_diff, axis=1)

    # ------------------------
    # Global latent positions: shape (N_NODES, K)
    # ------------------------
    # with numpyro.plate("nodes_global", N_NODES):
    #     u_loc = numpyro.sample(
    #         "u_loc",
    #         dist.MultivariateNormal(loc=jnp.zeros(K), covariance_matrix=jnp.eye(K)),
    #     )

    # ------------------------
    # Layer-specific offsets: shape (n_layers+1, N_NODES, K)
    # We give each layer's offsets a shrinkage prior, pushing them near 0.
    # ------------------------
    # with numpyro.plate("layers_offsets", n_layers + 1, dim=-2):
    #     with numpyro.plate("nodes_layer", N_NODES, dim=-1):
    #         U = numpyro.sample(
    #             "U",
    #             dist.MultivariateNormal(loc=jnp.zeros(K), covariance_matrix=jnp.eye(K)),
    #         )
    # Now U has shape (n_layers+1, N_NODES, K).
    # We build the actual position in layer ell as V[ell, i] = G[i] + U[ell, i].
    # We'll compute distances on the upper triangle index to match your triu_vals.
    # idx = jnp.triu_indices(N_NODES, k=1)

    # Distances for each layer (n_layers+1). We'll store them or compute them on the fly:
    # def layer_distance(ell):
    #     # V_ell shape (N_NODES, K)
    #     V_ell = u_loc + U[ell]  # broadcast addition
    #     # subtract positions
    #     diffs = V_ell[idx[0]] - V_ell[idx[1]]  # shape (#edges, K)
    #     dist_ell = jnp.linalg.norm(diffs, axis=1)
    #     return dist_ell

    # ------------------------
    # Likelihood for each *observed* layer
    # ------------------------
    # for ell in range(n_layers):
    #     # distance among nodes i, j in layer ell
    #     dist_ell = layer_distance(ell)
    #     # logit_ell = theta0[ell] - exp(theta1[ell]) * dist_ell
    #     logits = theta0[ell] - jnp.exp(theta1[ell]) * dist_ell
    #     numpyro.sample(f"obs_{ell}", dist.Bernoulli(logits=logits), obs=triu_vals[ell])

    # ------------------------
    # Likelihood for observed networks
    # ------------------------

    for k in range(n_layers):
        # logits_k = numpyro.deterministic(
        #     f"logits_{k}", theta0[k] - jnp.exp(theta1[k]) * V_norm
        # )
        logits_k = theta0[k] - jnp.exp(theta1[k]) * V_norm
        numpyro.sample(f"obs_{k}", dist.Bernoulli(logits=logits_k), obs=triu_vals[k])

    # ------------------------
    # Deterministic: Posterior p(A*_{ij} = 1)
    # (latent network edges, no likelihood)
    # ------------------------
    logits_latent = theta0[-1] - jnp.exp(theta1[-1]) * V_norm
    # dist_latent = layer_distance(n_layers)  # index n_layers => the *extra* layer
    # logits_latent = theta0[-1] - jnp.exp(theta1[-1]) * dist_latent
    numpyro.deterministic("probs_latent", jax.nn.sigmoid(logits_latent))


# --- Combined model ---


def combined_model(data, K=2):
    """
    Combined model for network (true and proxy) and outcome
    Used in the MWG sampler (for the continuous parameters)
    'triu_star' will be masked during continuous parameters sampling

    Args:
    data: dict with the following attributes:
        - Z: Treatment vector
        - Y: Outcome vector
        - triu_vals: n_layers x (n choose 2) array of edge values for observed networks
    """

    n_layers = data["triu_vals"].shape[0]

    # ------------------------
    # Hyper-priors (hierarchical)
    # ------------------------
    mu0 = numpyro.sample("mu0", dist.Normal(0.0, PRIOR_SCALE))
    mu1 = numpyro.sample("mu1", dist.Normal(0.0, PRIOR_SCALE))

    sigma0 = numpyro.sample("sigma0", dist.Gamma(2.0, 2.0))  # scale for theta0
    sigma1 = numpyro.sample("sigma1", dist.Gamma(2.0, 2.0))  # scale for theta1

    # We have n_layers+1 sets of (theta0_k, theta1_k):
    #   k=0,...,n_layers-1 for the observed networks
    #   k=n_layers for the *latent* (unobserved) network
    theta0 = []
    theta1 = []
    for k in range(n_layers + 1):
        theta0_k_nrm = numpyro.sample(f"theta0_{k}_nrm", dist.Normal())
        # theta0_k = numpyro.deterministic(f"theta0_{k}", theta0_k_nrm * sigma0 + mu0)
        theta0_k = theta0_k_nrm * sigma0 + mu0
        # t1_k = numpyro.sample(f"theta1_{k}", dist.Normal(mu1, sigma1))
        theta1_k_nrm = numpyro.sample(f"theta1_{k}_nrm", dist.Normal())
        # theta1_k = numpyro.deterministic(f"theta1_{k}", theta1_k_nrm * sigma1 + mu1)
        theta1_k = theta1_k_nrm * sigma1 + mu1
        # t1_k = t1_k * sigma1 + mu1
        theta0.append(theta0_k)
        theta1.append(theta1_k)

    # ------------------------
    # Shared latent positions
    # ------------------------
    # shape: (n, K)
    with numpyro.plate("nodes", N_NODES):
        V = numpyro.sample("V", dist.MultivariateNormal(0.0, jnp.eye(K)))

    idx = jnp.triu_indices(n=N_NODES, k=1)
    V_diff = V[idx[0]] - V[idx[1]]
    V_norm = jnp.linalg.norm(V_diff, axis=1)

    # ------------------------
    # Likelihood for observed networks
    # ------------------------

    for k in range(n_layers):
        logits_k = numpyro.deterministic(
            f"logits_{k}", theta0[k] - jnp.exp(theta1[k]) * V_norm
        )
        numpyro.sample(
            f"obs_{k}", dist.Bernoulli(logits=logits_k), obs=data["triu_vals"][k]
        )

    # ------------------------
    # Global latent positions: shape (N_NODES, K)
    # ------------------------
    # with numpyro.plate("nodes_global", N_NODES):
    #     u_loc = numpyro.sample(
    #         "u_loc",
    #         dist.MultivariateNormal(loc=jnp.zeros(K), covariance_matrix=jnp.eye(K)),
    #     )

    # ------------------------
    # Layer-specific offsets: shape (n_layers+1, N_NODES, K)
    # We give each layer's offsets a shrinkage prior, pushing them near 0.
    # ------------------------
    # with numpyro.plate("layers_offsets", n_layers + 1, dim=-2):
    #     with numpyro.plate("nodes_layer", N_NODES, dim=-1):
    #         U = numpyro.sample(
    #             "U",
    #             dist.MultivariateNormal(loc=jnp.zeros(K), covariance_matrix=jnp.eye(K)),
    #         )
    # Now U has shape (n_layers+1, N_NODES, K).
    # We build the actual position in layer ell as V[ell, i] = G[i] + U[ell, i].
    # We'll compute distances on the upper triangle index to match your triu_vals.
    # idx = jnp.triu_indices(N_NODES, k=1)

    # # Distances for each layer (n_layers+1). We'll store them or compute them on the fly:
    # def layer_distance(ell):
    #     # V_ell shape (N_NODES, K)
    #     V_ell = u_loc + U[ell]  # broadcast addition
    #     # subtract positions
    #     diffs = V_ell[idx[0]] - V_ell[idx[1]]  # shape (#edges, K)
    #     dist_ell = jnp.linalg.norm(diffs, axis=1)
    #     return dist_ell

    # ------------------------
    # Likelihood for each *observed* layer
    # ------------------------
    # for ell in range(n_layers):
    #     # distance among nodes i, j in layer ell
    #     dist_ell = layer_distance(ell)
    #     # logit_ell = theta0[ell] - exp(theta1[ell]) * dist_ell
    #     logits = theta0[ell] - jnp.exp(theta1[ell]) * dist_ell
    #     numpyro.sample(
    #         f"obs_{ell}", dist.Bernoulli(logits=logits), obs=data["triu_vals"][ell]
    #     )

    # ------------------------
    # Likelihood for latent network
    # ------------------------

    # dist_latent = layer_distance(n_layers)  # index n_layers => the *extra* layer
    # logits_star = numpyro.deterministic(
    # "logits_star", theta0[-1] - jnp.exp(theta1[-1]) * dist_latent
    # )
    logits_star = numpyro.deterministic(
        "logits_star", theta0[-1] - jnp.exp(theta1[-1]) * V_norm
    )
    triu_star = numpyro.sample("triu_star", dist.Bernoulli(logits=logits_star))

    # ------------------------
    # outcome model
    # ------------------------

    expos = ud.compute_exposures(triu_star, data["Z"])
    df_nodes = ud.get_df_nodes(data["Z"], expos)

    adj_mat = ud.triu_to_mat(triu_star) + jnp.eye(
        N_NODES
    )  # add self-loops for stability

    # priors

    with numpyro.plate("eta_plate", df_nodes.shape[1]):
        eta = numpyro.sample("eta", dist.Normal(0.0, PRIOR_SCALE))

    rho = numpyro.sample("rho", dist.Beta(2.0, 2.0))

    sig_inv = numpyro.sample("sig_inv", dist.Gamma(2.0, 2.0))

    # likelihood
    mean_y = df_nodes @ eta

    numpyro.sample(
        "Y",
        dist.CAR(
            loc=mean_y,
            correlation=rho,
            conditional_precision=sig_inv,
            adj_matrix=adj_mat,
        ),
        obs=data["Y"],
    )

    #  --- models for GWG ---


@jit
def cond_logpost_triu_star(triu_star, data, param):
    """
    Conditional log-posterior for A* given the rest of the parameters and data,

    p(A* | data, cotinuous parameters) \propto p(Y | Z, A*, eta) p(A* | theta, V)

    Args:
    triu_star: Upper triangular adjacency matrix
    data: Dict with the following attributes:
        - Z: Treatment vector
        - Y: Outcome vector
    param: dict with continuous parameters including:
        - eta: Coefficients for the outcome model
        - rho: Correlation parameter for the CAR model
        - sig_inv: Precision parameter for the CAR model
        - logits_star: Logits for the latent network edges

    """

    # prior network model
    a_star_logpmf = triu_star * jax.nn.log_sigmoid(param["logits_star"]) + (
        1 - triu_star
    ) * jax.nn.log_sigmoid(-param["logits_star"])

    # outcome model
    expos = ud.compute_exposures(triu_star, data["Z"])
    df_nodes = ud.get_df_nodes(data["Z"], expos)

    mean_y = df_nodes @ param["eta"]

    adj_mat = ud.triu_to_mat(triu_star) + jnp.eye(
        N_NODES
    )  # add self-loops for stability

    # CAR logdensity

    y_logpdf = models.car_logdensity(
        y=data["Y"],
        mu=mean_y,
        sigma=param["sig_inv"],
        rho=param["rho"],
        adj_matrix=adj_mat,
    )

    return a_star_logpmf.sum() + y_logpdf


# Gradient of A* conditional log-posterior
triu_star_grad_fn = jit(value_and_grad(cond_logpost_triu_star))
