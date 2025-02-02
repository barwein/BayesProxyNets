# Numpyro and manual models used in the simulations

import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import numpyro
import numpyro.distributions as dist
import src.utils as utils

### Numpyro models ###


def network_only_models_marginalized(data):
    """
    Model for network only models with marginalized A*
    Used in cut-posterior sampling

    Args:
    data: DataTuple object with the following attributes:
        - x_diff: Difference in x covaraiates
        - x2_or: if x2_i + x2_j = 1
        - triu_obs: Upper triangular observed adjacency matrix
    """
    # priors
    with numpyro.plate("theta_plate", 2):
        theta = numpyro.sample("theta", dist.Normal(0, 3))

    with numpyro.plate("gamma_plate", 3):
        gamma = numpyro.sample("gamma", dist.Normal(0, 3))

    # Calculate logits for A*
    star_logits = theta[0] + theta[1] * data.x2_or

    # Calculate logits for A_ij given A*_{ij} = 0
    obs_logits_k0 = gamma[1] + gamma[2] * data.x_diff

    # Compute log probs directly in log space for efficiency
    log_nu_k1 = star_logits - jnp.log1p(jnp.exp(star_logits))  # log sigmoid
    log_nu_k0 = -jnp.log1p(jnp.exp(star_logits))  # log(1-sigmoid)

    # Same for observation probs
    log_xi_k1 = data.triu_obs * gamma[0] - jnp.log1p(jnp.exp(gamma[0]))
    log_xi_k0 = data.triu_obs * obs_logits_k0 - jnp.log1p(jnp.exp(obs_logits_k0))

    # get A* posterior probs
    log_numerator = log_xi_k1 + log_nu_k1

    # denominator: sum_{k \in 0,1} xi(A_ij; k,gamma) * nu(k; theta)
    log_denominator = jnp.logaddexp(log_xi_k1 + log_nu_k1, log_xi_k0 + log_nu_k0)
    # p(A* | \theta, \gamma, data)
    astar_probs = jnp.exp(log_numerator - log_denominator)
    numpyro.deterministic("triu_star_probs", astar_probs)

    # likelihood term
    numpyro.factor("marginalized_likelihood", log_denominator.sum())


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
        jnp.stack([jnp.ones(data.Y.shape[0]), data.Z, data.X, expos])
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
    """Compute log density of CAR model

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


# Gradient of the conditional log-posterior for A*
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
