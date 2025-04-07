# Numpyro and manual models used in the simulations

import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import jax
import numpyro
import numpyro.distributions as dist
import src.utils as utils


### --- Probabilistic models --- ###

# True network: p(A*|X,\theta) \propto expit(theta_0 + theta_1 * x2_or)
# Proxy network: p(A_ij|A*_{ij},X,\gamma) \propto expit(gamma_0 * A*_{ij} + (1-A*_{ij}) * (gamma_1 + gamma_2 * x_diff))
# Proxy network (rep): P(A^r_ij|A*_{ij}, A_{ij}, X, \gamma)
# Outcome: p(Y|A*,X,Z,\eta,\rho,\sig_inv) \propto N(Y|mu; Sigma)
# where mu = df @ eta, with df = [1, Z, X, exposures] and exposures = compute_exposures(A*, Z) (weighted sum of Z by deg cen)
# and Sigma = [sig_inv(D - rho * A*]^{-1}, with D=diag(degrees(A*)) --> CAR model

# priors are:
# theta ~ N(0, 3)
# gamma ~ N(0, 3)
# eta ~ N(0, 3)
# rho ~ Beta(2, 2)
# sig_inv ~ Gamma(2, 2)

# PRIOR_SCALE = jnp.sqrt(3).item()
# PRIOR_SCALE = jnp.sqrt(5).item()
PRIOR_SCALE = 3

### --- NumPyro models --- ###

# cut-posterior models


def networks_marginalized_model(data):
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
    theta = numpyro.sample("theta", dist.Normal(0, PRIOR_SCALE).expand([2]))
    gamma = numpyro.sample("gamma", dist.Normal(0, PRIOR_SCALE).expand([3]))

    # P(A*_ij=1)
    star_probs = jax.nn.sigmoid(theta[0] + theta[1] * data.x2_or)
    star_probs = jnp.clip(star_probs, 1e-6, 1 - 1e-6)

    # P(A_ij = 1 | A*_ij = 0)
    obs_probs_k0 = jax.nn.sigmoid(gamma[1] + gamma[2] * data.x_diff)
    obs_probs_k0 = jnp.clip(obs_probs_k0, 1e-6, 1 - 1e-6)

    # P(A_ij = 1 | A*_ij = 1)
    obs_probs_k1 = jax.nn.sigmoid(gamma[0])
    obs_probs_k1 = jnp.clip(obs_probs_k1, 1e-6, 1 - 1e-6)

    # marginalized probs P(A_ij=1)
    mixed_probs = star_probs * obs_probs_k1 + (1 - star_probs) * obs_probs_k0

    with numpyro.plate("edges", data.triu_obs.shape[0]):
        numpyro.sample("obs", dist.BernoulliProbs(mixed_probs), obs=data.triu_obs)

    # save posterior A_star probs
    # let  p_1 = P(A*_ij=1)*P(A_ij| A*_ij=1)
    #      p_0 = P(A*_ij=0 )*P(A_ij| A*_ij=0)
    # then posterior probs P(A*_ij | A, \theta,\ gamma) = p_1 / (p_1 + p_0)

    # numerator aka p_1
    numerator = jnp.where(
        data.triu_obs == 1.0, star_probs * obs_probs_k1, star_probs * (1 - obs_probs_k1)
    )
    # denominator aka p_1 + p_0
    denominator = numerator + jnp.where(
        data.triu_obs == 1.0,
        (1 - star_probs) * obs_probs_k0,
        (1 - star_probs) * (1 - obs_probs_k0),
    )

    numpyro.deterministic("triu_star_probs", numerator / denominator)


def networks_marginalized_model_rep(data):
    """
    Model for network only models with marginalized A*
    Used in cut-posterior sampling

    We have repeated proxy measures A^r, A

    We embed it as mixture model of (A^r, A) edges (4 categories)

    Args:
      data: an object with attributes:
           - x_diff: a 1D array (one per edge) of differences in covariates,
           - x2_or: a 1D array of indicators (one per edge) for whether x2_i + x2_j = 1,
           - triu_obs: a 1D array of observed edge indicators (binary; one per edge).
           - triu_obs_rep: a 1D array of repeated observed edge indicators (binary; one per edge).
    """

    # priors
    theta = numpyro.sample("theta", dist.Normal(0, PRIOR_SCALE).expand([2]))
    gamma = numpyro.sample("gamma", dist.Normal(0, PRIOR_SCALE).expand([7]))

    # P(A*_ij=1)
    star_probs = jax.nn.sigmoid(theta[0] + theta[1] * data.x2_or)
    star_probs = jnp.clip(star_probs, 1e-6, 1 - 1e-6)

    # P(A_ij = 1 | A*_ij = 1)
    obs_probs_k1 = jax.nn.sigmoid(gamma[0])
    obs_probs_k1 = jnp.clip(obs_probs_k1, 1e-6, 1 - 1e-6)

    # P(A_ij = 1 | A*_ij = 0)
    obs_probs_k0 = jax.nn.sigmoid(gamma[1] + gamma[2] * data.x_diff)
    obs_probs_k0 = jnp.clip(obs_probs_k0, 1e-6, 1 - 1e-6)

    # P(A^r_ij=1 | A*_ij=1, A_ij)
    obs_rep_probs_k1 = jax.nn.sigmoid(gamma[3] + gamma[4] * data.triu_obs)
    obs_rep_probs_k1 = jnp.clip(obs_rep_probs_k1, 1e-6, 1 - 1e-6)

    # P(A^r_ij=1 | A*_ij=0, A_ij)
    obs_rep_probs_k0 = jax.nn.sigmoid(gamma[5] + gamma[6] * data.triu_obs)
    obs_rep_probs_k0 = jnp.clip(obs_rep_probs_k0, 1e-6, 1 - 1e-6)

    # compute marginalize probs of (A^_ij=r, A_ij=a), a,r \in {0,1}
    # represent as categorical $C_ij = 2 A_ij + A^r_ij \in {0,1,2,3}
    # Given A*_ij=1
    # p(A_ij=r, A^r_ij=a | A*_ij=1)*P(A*_ij=1)
    pj_star1_cat0 = star_probs * (1 - obs_probs_k1) * (1 - obs_rep_probs_k1)
    pj_star1_cat1 = star_probs * (1 - obs_probs_k1) * obs_rep_probs_k1
    pj_star1_cat2 = star_probs * obs_probs_k1 * (1 - obs_rep_probs_k1)
    pj_star1_cat3 = star_probs * obs_probs_k1 * obs_rep_probs_k1

    # Given A*_ij=0
    # p(A_ij=r, A^r_ij=a | A*_ij=0)*P(A*_ij=0)
    pj_star0_cat0 = (1 - star_probs) * (1 - obs_probs_k0) * (1 - obs_rep_probs_k0)
    pj_star0_cat1 = (1 - star_probs) * (1 - obs_probs_k0) * obs_rep_probs_k0
    pj_star0_cat2 = (1 - star_probs) * obs_probs_k0 * (1 - obs_rep_probs_k0)
    pj_star0_cat3 = (1 - star_probs) * obs_probs_k0 * obs_rep_probs_k0

    # marginalized probs P(C_ij=c) = \sum_{0,1} p(A_ij=r, A^r_ij=a | A*_ij=k)*P(A*_ij=k)
    p_cat0 = pj_star1_cat0 + pj_star0_cat0
    p_cat1 = pj_star1_cat1 + pj_star0_cat1
    p_cat2 = pj_star1_cat2 + pj_star0_cat2
    p_cat3 = pj_star1_cat3 + pj_star0_cat3

    probs = jnp.stack([p_cat0, p_cat1, p_cat2, p_cat3], axis=-1)
    probs = probs / jnp.sum(probs, axis=-1, keepdims=True)

    # observed categorical data
    obs_cat = jnp.astype(2 * data.triu_obs + data.triu_obs_rep, jnp.int32)

    with numpyro.plate("edges", data.triu_obs.shape[0]):
        numpyro.sample("obs_joint", dist.Categorical(probs=probs), obs=obs_cat)

    # --- Compute the posterior probability of A*_ij=1 given the joint observation ---
    # Select the appropriate terms based on the observed category
    # p_1 = p(C_ij = c | A*_ij = 1)p(A*_ij = 1)
    numerator = jnp.where(
        obs_cat == 0,
        pj_star1_cat0,
        jnp.where(
            obs_cat == 1,
            pj_star1_cat1,
            jnp.where(obs_cat == 2, pj_star1_cat2, pj_star1_cat3),
        ),
    )

    # p_0 = p(C_ij = c | A*_ij = 0)p(A*_ij = 0)
    # denom is p_0 + p_1
    denominator = numerator + jnp.where(
        obs_cat == 0,
        pj_star0_cat0,
        jnp.where(
            obs_cat == 1,
            pj_star0_cat1,
            jnp.where(obs_cat == 2, pj_star0_cat2, pj_star0_cat3),
        ),
    )
    # p_1 / (p_0 + p_1)
    posterior_star = numerator / denominator

    # Save the posterior probabilities for A* (for example, on the upper triangle of the network)
    numpyro.deterministic("triu_star_probs", posterior_star)


# def plugin_outcome_model(df_nodes, adj_mat, Y):
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
        eta = numpyro.sample("eta", dist.Normal(0, PRIOR_SCALE))

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

    # with numpyro.plate("Y plate", Y.shape[0]):
    #     numpyro.sample("Y", dist.Normal(mean_y, sig_inv), obs=Y)


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
        theta = numpyro.sample("theta", dist.Normal(0, PRIOR_SCALE))

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
        eta = numpyro.sample("eta", dist.Normal(0, PRIOR_SCALE))

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
    # with numpyro.plate("Y plate", data.Y.shape[0]):
    #     numpyro.sample("Y", dist.Normal(mean_y, sig_inv), obs=data.Y)

    # === Proxy network model with single measures ===
    # priors
    with numpyro.plate("gamma_plate", 3):
        gamma = numpyro.sample("gamma", dist.Normal(0, PRIOR_SCALE))

    # likelihood
    obs_logits = triu_star * gamma[0] + (1 - triu_star) * (
        gamma[1] + gamma[2] * data.x_diff
    )

    with numpyro.plate("edges", data.triu_obs.shape[0]):
        numpyro.sample("triu_obs", dist.Bernoulli(logits=obs_logits), obs=data.triu_obs)


def combined_model_rep(data):
    """
    Combined model for network (true and proxy) and outcome
    Used in the MWG sampler (for the continuous parameters)
    'triu_star' will be masked during continuous parameters sampling

    Repeated proxy measurement A^r and A

    Args:
    data: DataTuple object with the following attributes:
        - x_diff: Difference in x covaraiates
        - x2_or: if x2_i + x2_j = 1
        - triu_obs: Upper triangular observed adjacency matrix
        - triu_obs_rep: Repeated Upper triangular observed adjacency matrix
        - Z: Treatment vector
        - X: Node-level covariate
        - Y: Outcome vector
    """
    # === True network model ===
    # priors
    with numpyro.plate("theta_plate", 2):
        theta = numpyro.sample("theta", dist.Normal(0, PRIOR_SCALE))

    # likelihood
    star_logits = theta[0] + theta[1] * data.x2_or
    triu_star = numpyro.sample("triu_star", dist.Bernoulli(logits=star_logits))

    adj_mat = utils.Triu_to_mat(triu_star)

    # === Outcome model ===
    expos = utils.compute_exposures(triu_star, data.Z)
    df_nodes = jnp.transpose(
        jnp.stack([jnp.ones(data.Z.shape[0]), data.Z, data.x, expos])
    )

    # priors
    with numpyro.plate("eta_plate", df_nodes.shape[1]):
        eta = numpyro.sample("eta", dist.Normal(0, PRIOR_SCALE))

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

    # with numpyro.plate("Y plate", data.Y.shape[0]):
    #     numpyro.sample("Y", dist.Normal(mean_y, sig_inv), obs=data.Y)

    # === Proxy network model with repeated measures ===
    # priors
    with numpyro.plate("gamma_plate", 7):
        gamma = numpyro.sample("gamma", dist.Normal(0, PRIOR_SCALE))

    # First proxy measurement (A)
    logit_A = jnp.where(triu_star == 1, gamma[0], gamma[1] + gamma[2] * data.x_diff)

    # Second proxy measurement (A^r)
    logit_A_rep = jnp.where(
        triu_star == 1,
        gamma[3] + gamma[4] * data.triu_obs,
        gamma[5] + gamma[6] * data.triu_obs,
    )

    # likelihood
    with numpyro.plate("edges", data.triu_obs.shape[0]):
        numpyro.sample("triu_obs", dist.Bernoulli(logits=logit_A), obs=data.triu_obs)
        numpyro.sample(
            "triu_obs_rep", dist.Bernoulli(logits=logit_A_rep), obs=data.triu_obs_rep
        )


### --- Manual log-densities models --- ###


@jit
def car_logdensity(y, mu, sigma, rho, adj_matrix):
    """
    Compute log density of CAR model using sparse representation
    https://mc-stan.org/learn-stan/case-studies/mbjoseph-CARStan.html

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
def iid_logdensity(y, mu, sigma):
    y_centered = y - mu
    n = y.shape[0]

    normalizing = -n / 2 * jnp.log(2 * jnp.pi)
    sig_term = -n * jnp.log(sigma)

    quad_term = (-0.5 / jnp.square(sigma)) * jnp.sum(y_centered**2)

    return normalizing + sig_term + quad_term


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
    # a_star_logpmf = triu_star * logits_a_star - jnp.log1p(jnp.exp(logits_a_star))
    a_star_logpmf = triu_star * jax.nn.log_sigmoid(logits_a_star) + (
        1 - triu_star
    ) * jax.nn.log_sigmoid(-logits_a_star)

    # p(A|A*,X,\gamma)
    logits_a_obs = triu_star * param.gamma[0] + (1 - triu_star) * (
        param.gamma[1] + param.gamma[2] * data.x_diff
    )
    a_obs_logpmf = data.triu_obs * jax.nn.log_sigmoid(logits_a_obs) + (
        1 - data.triu_obs
    ) * jax.nn.log_sigmoid(-logits_a_obs)

    # p(Y|A*,X,Z,\eta,\sig_y)
    exposures = utils.compute_exposures(triu_star, data.Z)
    df_nodes = jnp.transpose(
        jnp.stack([jnp.ones(data.Y.shape[0]), data.Z, data.x, exposures])
    )
    mean_y = jnp.dot(df_nodes, param.eta)

    # y_logpdf = iid_logdensity(data.Y, mean_y, param.sig_inv)

    y_logpdf = car_logdensity(
        data.Y, mean_y, param.sig_inv, param.rho, utils.Triu_to_mat(triu_star)
    )

    return a_star_logpmf.sum() + a_obs_logpmf.sum() + y_logpdf


# Gradient of A* conditional log-posterior
triu_star_grad_fn = jit(value_and_grad(cond_logpost_a_star))


@jit
def cond_logpost_a_star_rep(triu_star, data, param):
    """
    Conditional log-posterior for A* given the rest of the parameters and data,
    Including repeated proxy measures A^r and A

    Args:
    triu_star: Upper triangular adjacency matrix
    data: DataTuple object with the following attributes:
        - x_diff: Difference in x covaraiates
        - x2_or: if x2_i + x2_j = 1
        - triu_obs: Upper triangular observed adjacency matrix
        - triu_obs_rep: Repeated Upper triangular observed adjacency matrix
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
    # a_star_logpmf = triu_star * logits_a_star - jnp.log1p(jnp.exp(logits_a_star))
    a_star_logpmf = triu_star * jax.nn.log_sigmoid(logits_a_star) + (
        1 - triu_star
    ) * jax.nn.log_sigmoid(-logits_a_star)

    # p(A|A*,X,\gamma)
    logits_a_obs = triu_star * param.gamma[0] + (1 - triu_star) * (
        param.gamma[1] + param.gamma[2] * data.x_diff
    )
    a_obs_logpmf = data.triu_obs * jax.nn.log_sigmoid(logits_a_obs) + (
        1 - data.triu_obs
    ) * jax.nn.log_sigmoid(-logits_a_obs)

    # p(A^r | A, A*, gamma) for the repeated measurement.
    logit_a_obs_rep = jnp.where(
        triu_star == 1,
        param.gamma[3] + param.gamma[4] * data.triu_obs,  # when A* == 1
        param.gamma[5] + param.gamma[6] * data.triu_obs,  # when A* == 0
    )

    a_obs_rep_logpmf = data.triu_obs_rep * jax.nn.log_sigmoid(logit_a_obs_rep) + (
        1 - data.triu_obs_rep
    ) * jax.nn.log_sigmoid(-logit_a_obs_rep)

    # p(Y|A*,X,Z,\eta,\sig_y)
    exposures = utils.compute_exposures(triu_star, data.Z)
    df_nodes = jnp.transpose(
        jnp.stack([jnp.ones(data.Y.shape[0]), data.Z, data.x, exposures])
    )
    mean_y = jnp.dot(df_nodes, param.eta)

    # y_logpdf = iid_logdensity(data.Y, mean_y, param.sig_inv)

    y_logpdf = car_logdensity(
        data.Y, mean_y, param.sig_inv, param.rho, utils.Triu_to_mat(triu_star)
    )

    return a_star_logpmf.sum() + a_obs_logpmf.sum() + a_obs_rep_logpmf.sum() + y_logpdf


# Gradient of A* conditional log-posterior
triu_star_grad_fn_rep = jit(value_and_grad(cond_logpost_a_star_rep))


def compute_log_cut_posterior(astar_sample, theta, gamma, data):
    """
    Compute log cut-posterior of $A*|theta,gamma$ in network module (true and proxy)
    for a single astar configuration

    single proxy network

    Args:
    astar_sample: A* sample
    theta: Parameters for A*
    gamma: Parameters for A_ij
    data: DataTuple object
    """
    # Prior term (log p(A*|theta))
    star_logits = theta[0] + theta[1] * data.x2_or
    log_prior = astar_sample * jax.nn.log_sigmoid(star_logits) + (
        1 - astar_sample
    ) * jax.nn.log_sigmoid(-star_logits)

    # Likelihood term (log p(A|A*,gamma))
    obs_logits = astar_sample * gamma[0] + (1 - astar_sample) * (
        gamma[1] + gamma[2] * data.x_diff
    )
    log_lik = data.triu_obs * jax.nn.log_sigmoid(obs_logits) + (
        1 - data.triu_obs
    ) * jax.nn.log_sigmoid(-obs_logits)

    return jnp.sum(log_prior + log_lik)


# Vectorize over samples
compute_log_posterior_vmap = vmap(
    compute_log_cut_posterior, in_axes=(0, None, None, None)
)


def compute_log_cut_posterior_rep(astar_sample, theta, gamma, data):
    """
    Compute log cut-posterior of $A*|theta,gamma$ in network module (true and proxy)
    for a single astar configuration

    repeated proxy networks measures

    Args:
    astar_sample: A* sample
    theta: Parameters for A*
    gamma: Parameters for A_ij
    data: DataTuple object
    """
    # Prior term (log p(A*|theta))
    star_logits = theta[0] + theta[1] * data.x2_or
    log_prior = astar_sample * jax.nn.log_sigmoid(star_logits) + (
        1 - astar_sample
    ) * jax.nn.log_sigmoid(-star_logits)

    # Likelihood term (log p(A|A*,gamma))
    obs_logits = astar_sample * gamma[0] + (1 - astar_sample) * (
        gamma[1] + gamma[2] * data.x_diff
    )
    log_lik = data.triu_obs * jax.nn.log_sigmoid(obs_logits) + (
        1 - data.triu_obs
    ) * jax.nn.log_sigmoid(-obs_logits)

    # Likelihood term (log p(A^r|A,A*,gamma))
    logit_A_rep = jnp.where(
        astar_sample == 1,
        gamma[3] + gamma[4] * data.triu_obs,
        gamma[5] + gamma[6] * data.triu_obs,
    )
    log_lik_rep = data.triu_obs_rep * jax.nn.log_sigmoid(logit_A_rep) + (
        1 - data.triu_obs_rep
    ) * jax.nn.log_sigmoid(-logit_A_rep)

    return jnp.sum(log_prior + log_lik + log_lik_rep)


# Vectorize over samples
compute_log_posterior_vmap_rep = vmap(
    compute_log_cut_posterior_rep, in_axes=(0, None, None, None)
)
