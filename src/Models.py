# Numpyro and manual models used in the simulations

import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import jax
import numpyro
import numpyro.distributions as dist
import src.utils as utils
from numpyro.distributions.util import clamp_probs


### --- Probabilistic models --- ###

# True network: p(A*|X,\theta) \propto expit(theta_0 + theta_1 * x2_or)
# Proxy network: p(A_ij|A*_{ij},X,\gamma) \propto expit(gamma_0 * A*_{ij} + (1-A*_{ij}) * (gamma_1 + gamma_2 * x_diff))
# TODO: Proxy network (rep): p(A^{rep}_{ij} | )
# Outcome: p(Y|A*,X,Z,\eta,\rho,\sig_inv) \propto N(Y|mu; Sigma)
# where mu = df @ eta, with df = [1, Z, X, exposures] and exposures = compute_exposures(A*, Z) (weighted sum of Z by deg cen)
# and Sigma = [sig_inv(D - rho * A*]^{-1}, with D=diag(degrees(A*)) --> CAR model

# priors are:
# theta ~ N(0, 3)
# gamma ~ N(0, 3)
# eta ~ N(0, 3)
# rho ~ Beta(2, 2)
# sig_inv ~ Gamma(2, 2)


### Numpyro models ###


def network_only_models_marginalized(data):
    """
    Model for network only models with marginalized A*
    Used in cut-posterior sampling

    We embed it as mixture model of A edges

    Args:
      data: an object with attributes:
           - x_diff: a 1D array (one per edge) of differences in covariates,
           - x2_or: a 1D array of indicators (one per edge) for whether x2_i + x2_j = 1,
           - triu_obs: a 1D array of observed edge indicators (binary; one per edge).
    """
    # --- Priors ---
    # theta: parameters for the latent network A*
    theta = numpyro.sample("theta", dist.Normal(0, 3).expand([2]))
    # gamma: parameters for the edge likelihood given A*
    gamma = numpyro.sample("gamma", dist.Normal(0, 3).expand([3]))

    # --- Latent network A* prior embedded in the mixture weights ---
    # Compute the logits for the latent A* for each edge.
    star_logits = theta[0] + theta[1] * data.x2_or
    star_logits = jnp.clip(star_logits, -20, 20)  # avoid overflow

    prior_Astar_1 = jax.nn.sigmoid(star_logits)
    # Mixture probabilities: first component corresponds to A* = 0, second to A* = 1.
    mix_probs = jnp.stack([1 - prior_Astar_1, prior_Astar_1], axis=-1)

    # # The probability that A* = 1 is sigmoid(star_logits)
    # # Create a 2-element vector of mixture weights for each edge:
    # # component 0 (A* = 0) gets weight 1 - sigmoid(star_logits)
    # # component 1 (A* = 1) gets weight sigmoid(star_logits)
    # mix_probs = jnp.stack([1 - jax.nn.sigmoid(star_logits),
    #                        jax.nn.sigmoid(star_logits)], axis=-1)

    # --- Likelihood for A given A* ---
    # When A* = 0, we use a logistic regression with linear predictor:
    obs_logits_k0 = gamma[1] + gamma[2] * data.x_diff
    obs_logits_k0 = jnp.clip(obs_logits_k0, -20, 20)  # avoid overflow

    # When A* = 1, the likelihood is given by a fixed Bernoulli with logit gamma[0].
    comp_logits = jnp.stack(
        [obs_logits_k0, gamma[0] * jnp.ones_like(obs_logits_k0)], axis=-1
    )

    # --- Define the mixture distribution ---
    # The mixture is over two Bernoulli components.
    mixture_dist = dist.MixtureSameFamily(
        # The categorical mixing distribution with probabilities given per edge.
        mixing_distribution=dist.Categorical(probs=mix_probs),
        # The two components: Bernoulli with logits defined in comp_logits.
        component_distribution=dist.Bernoulli(logits=comp_logits),
    )

    # --- Observation ---
    # Each edge is conditionally independent.
    with numpyro.plate("edges", data.triu_obs.shape[0]):
        numpyro.sample("obs", mixture_dist, obs=data.triu_obs)

    # --- Compute the posterior probabilities for A* ---
    # Compute the log-likelihoods for each component for each edge.
    # For component 0 (A* = 0):
    log_lik_0 = data.triu_obs * obs_logits_k0 - jax.nn.softplus(obs_logits_k0)
    # For component 1 (A* = 1):
    log_lik_1 = data.triu_obs * gamma[0] - jax.nn.softplus(gamma[0])

    # Incorporate the log prior probabilities:
    log_prior_0 = jnp.log(1 - prior_Astar_1)
    log_prior_1 = jnp.log(prior_Astar_1)

    log_joint_0 = log_prior_0 + log_lik_0
    log_joint_1 = log_prior_1 + log_lik_1

    # Normalizing constant per edge:
    log_norm = jnp.logaddexp(log_joint_0, log_joint_1)
    # Posterior probability for A* = 1:
    astar_probs = jnp.exp(log_joint_1 - log_norm)

    # Return as a deterministic output (for each edge)
    numpyro.deterministic("triu_star_probs", astar_probs)


# def network_only_models_marginalized(data):
#     """
#     Model for network only models with marginalized A*
#     Used in cut-posterior sampling

#     Args:
#     data: DataTuple object with the following attributes:
#         - x_diff: Difference in x covaraiates
#         - x2_or: if x2_i + x2_j = 1
#         - triu_obs: Upper triangular observed adjacency matrix
#     """
#     # priors
#     # with numpyro.plate("theta_plate", 2):
#     #     theta = numpyro.sample("theta", dist.Normal(0, 5))

#     # with numpyro.plate("gamma_plate", 3):
#     #     gamma = numpyro.sample("gamma", dist.Normal(0, 5))
#     theta = numpyro.sample("theta", dist.Normal(0, 5).expand([2]))
#     gamma = numpyro.sample("gamma", dist.Normal(0, 5).expand([3]))

#     # Calculate logits for A*
#     star_logits = theta[0] + theta[1] * data.x2_or
#     star_logits = jnp.clip(star_logits, -20, 20)  # Avoid overflow

#     # Calculate logits for A_ij given A*_{ij} = 0
#     obs_logits_k0 = gamma[1] + gamma[2] * data.x_diff
#     obs_logits_k0 = jnp.clip(obs_logits_k0, -20, 20)  # Avoid overflow

#     # Compute log probs directly in log space for efficiency
#     # log_nu_k1 = star_logits - jnp.log1p(jnp.exp(star_logits))  # log sigmoid
#     log_nu_k1 = star_logits - jax.nn.softplus(star_logits)  # log sigmoid
#     # log_nu_k0 = -jnp.log1p(jnp.exp(star_logits))  # log(1-sigmoid)
#     log_nu_k0 = -jax.nn.softplus(star_logits)  # log(1-sigmoid)

#     # Same for observation probs
#     # log_xi_k1 = data.triu_obs * gamma[0] - jnp.log1p(jnp.exp(gamma[0]))
#     log_xi_k1 = data.triu_obs * gamma[0] - jax.nn.softplus(gamma[0])
#     # log_xi_k0 = data.triu_obs * obs_logits_k0 - jnp.log1p(jnp.exp(obs_logits_k0))
#     log_xi_k0 = data.triu_obs * obs_logits_k0 - jax.nn.softplus(obs_logits_k0)

#     # get A* posterior probs
#     # log_numerator = log_xi_k1 + log_nu_k1
#     log_joint_1 = log_xi_k1 + log_nu_k1
#     log_joint_0 = log_xi_k0 + log_nu_k0

#     # denominator: sum_{k \in 0,1} xi(A_ij; k,gamma) * nu(k; theta)
#     # log_denominator = jnp.logaddexp(log_xi_k1 + log_nu_k1, log_xi_k0 + log_nu_k0)
#     log_denominator = jnp.logaddexp(log_joint_1, log_joint_0)

#     # likelihood term
#     # numpyro.factor("marginalized_likelihood", log_denominator.sum())
#     with numpyro.plate("edges", data.triu_obs.shape[0]):
#         numpyro.factor("marginalized_likelihood", log_denominator)

#     # p(A* | \theta, \gamma, data)
#     # astar_probs = jnp.exp(log_numerator - log_denominator)
#     astar_probs = jnp.exp(log_joint_1 - log_denominator)
#     numpyro.deterministic("triu_star_probs", astar_probs)


def plugin_outcome_model(df_nodes, adj_mat, Y):
    """
    Plugin model for outcome given A*
    Used in cut-posterior sampling

    Args:
    df_nodes: Node-level covariates (including exposures)
    adj_mat: A* triu values
    Y: Outcome vector
    """

    # priors
    with numpyro.plate("eta_plate", df_nodes.shape[1]):
        eta = numpyro.sample("eta", dist.Normal(0, 3))

    rho = numpyro.sample("rho", dist.Beta(2, 2))

    sig_inv = numpyro.sample("sig_inv", dist.Gamma(2, 2))

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
        obs=Y,
    )


def combined_model(data):
    """
    Combined model for network (true and proxy) and outcome
    Used in the MWG sampler (for the continuous parameters)
    'triu_star' will be masked during continuous parameters sampling

    Args:
    data: DataTuple object with the following attributes:
        - x_diff: Difference in x covaraiates
        - x2_or: if x2_i + x2_j = 1
        - triu_obs: Upper triangular observed adjacency matrix
        - Z: Treatment vector
        - X: Node-level covariate
        - Y: Outcome vector
    """
    # True network model
    # priors
    with numpyro.plate("theta_plate", 2):
        theta = numpyro.sample("theta", dist.Normal(0, 3))

    # likelihood
    star_logits = theta[0] + theta[1] * data.x2_or
    triu_star = numpyro.sample("triu_star", dist.Bernoulli(logits=star_logits))
    adj_mat = utils.Triu_to_mat(triu_star)

    # Outcome model
    expos = utils.compute_exposures(triu_star, data.Z)
    df_nodes = jnp.transpose(
        jnp.stack([jnp.ones(data.Z.shape[0]), data.Z, data.x, expos])
    )

    # priors
    with numpyro.plate("eta_plate", df_nodes.shape[1]):
        eta = numpyro.sample("eta", dist.Normal(0, 3))

    rho = numpyro.sample("rho", dist.Beta(2, 2))

    sig_inv = numpyro.sample("sig_inv", dist.Gamma(2, 2))

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
        obs=data.Y,
    )

    # Proxy nets model
    # priors
    with numpyro.plate("gamma_plate", 3):
        gamma = numpyro.sample("gamma", dist.Normal(0, 3))

    # likelihood
    obs_logits = triu_star * gamma[0] + (1 - triu_star) * (
        gamma[1] + gamma[2] * data.x_diff
    )

    numpyro.sample("triu_obs", dist.Bernoulli(logits=obs_logits), obs=data.triu_obs)


### Manual models ###


@jit
def car_logdensity(y, mu, sigma, rho, adj_matrix):
    """
    Compute log density of CAR model

    Args:
        y: array of shape (n,) - observed values
        mu: array of shape (n,) - mean parameter
        sigma: float - conditional precision parameter (positive)
        rho: float - correlation parameter in [0,1]
        adj_matrix: array of shape (n,n) - symmetric adjacency matrix

    Returns:
        float: log density value
    """
    # Center the observations
    y_centered = y - mu

    # Compute degree matrix diagonal
    D = adj_matrix.sum(axis=-1)
    # replace D==0 with 1.0.
    D = jnp.where(D == 0, 1.0, D)
    D_rsqrt = D ** (-0.5)

    # Compute normalized adjacency matrix for eigenvalues
    adj_scaled = adj_matrix * (D_rsqrt[:, None] * D_rsqrt[None, :])

    # Get eigenvalues of normalized adjacency
    lam = jnp.linalg.eigvalsh(adj_scaled)

    n = y.shape[0]

    # Compute components of log density
    logprec = n * jnp.log(sigma)
    logdet = jnp.sum(jnp.log1p(-rho * lam)) + jnp.sum(jnp.log(D))

    quad = sigma * (
        jnp.sum(D * y_centered**2)
        - rho * jnp.sum(y_centered * (adj_matrix @ y_centered))
    )

    return 0.5 * (-n * jnp.log(2 * jnp.pi) + logprec + logdet - quad)


@jit
def cond_logpost_a_star(triu_star, data, param):
    """
    Conditional log-posterior for A* given the rest of the parameters and data

    Args:
    triu_star: Upper triangular adjacency matrix
    data: DataTuple object with the following attributes:
        - x_diff: Difference in x covaraiates
        - x2_or: if x2_i + x2_j = 1
        - triu_obs: Upper triangular observed adjacency matrix
        - Z: Treatment vector
        - x: Node-level covariate
        - Y: Outcome vector
    param: ParamTuple object with the following attributes:
        - theta: Parameters for A*
        - gamma: Parameters for A_ij
        - eta: Parameters for outcome model
        - rho: Correlation parameter
        - sig_inv: Conditional precision parameter

    """
    # p(A*|X,\theta)
    logits_a_star = param.theta[0] + param.theta[1] * data.x2_or
    a_star_logpmf = triu_star * logits_a_star - jnp.log1p(jnp.exp(logits_a_star))

    # p(A|A*,X,\gamma)
    logits_a_obs = (triu_star * param.gamma[0]) + (1 - triu_star) * (
        param.gamma[1] + param.gamma[2] * data.x_diff
    )
    a_obs_logpmf = data.triu_obs * logits_a_obs - jnp.log1p(jnp.exp(logits_a_obs))

    # p(Y|A*,X,Z,\eta,\sig_y)
    exposures = utils.compute_exposures(triu_star, data.Z)
    df_nodes = jnp.transpose(
        jnp.stack([jnp.ones(data.Y.shape[0]), data.Z, data.x, exposures])
    )
    mean_y = jnp.dot(df_nodes, param.eta)

    y_logpdf = car_logdensity(
        data.Y, mean_y, param.sig_inv, param.rho, utils.Triu_to_mat(triu_star)
    )

    return a_star_logpmf.sum() + a_obs_logpmf.sum() + y_logpdf


# Gradient of A* conditional log-posterior
triu_star_grad_fn = jit(value_and_grad(cond_logpost_a_star))


def compute_log_cut_posterior(astar_sample, theta, gamma, data):
    """
    Compute log cut-posterior of $A*|theta,gamma$ in network module (true and proxy)
    for a single astar configuration

    Args:
    astar_sample: A* sample
    theta: Parameters for A*
    gamma: Parameters for A_ij
    data: DataTuple object
    """
    # Prior term (log p(A*|theta))
    star_logits = theta[0] + theta[1] * data.x2_or
    log_prior = jnp.where(
        astar_sample == 1,
        star_logits - jnp.log1p(jnp.exp(star_logits)),  # log p(A*=1)
        -jnp.log1p(jnp.exp(star_logits)),
    )  # log p(A*=0)

    # Likelihood term (log p(A|A*,gamma))
    obs_logits_k0 = gamma[1] + gamma[2] * data.x_diff
    log_lik = jnp.where(
        astar_sample == 1,
        data.triu_obs * gamma[0] - jnp.log1p(jnp.exp(gamma[0])),  # when A*=1
        data.triu_obs * obs_logits_k0 - jnp.log1p(jnp.exp(obs_logits_k0)),
    )  # when A*=0

    return jnp.sum(log_prior + log_lik)


# Vectorize over samples
compute_log_posterior_vmap = vmap(
    compute_log_cut_posterior, in_axes=(0, None, None, None)
)
