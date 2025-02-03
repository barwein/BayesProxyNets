###
# Functions that generate data for simulations
###

import jax.numpy as jnp
from jax.scipy.special import expit
import numpy as np
import src.utils as utils

# --- Data generation functions ---


def generate_covariates(rng, n):
    x = jnp.array(rng.normal(loc=0, scale=1, size=n), dtype=jnp.float32)
    x2 = jnp.array(rng.binomial(n=1, p=0.1, size=n), dtype=jnp.float32)
    return x, x2


def compute_pairwise_diffs(n, x, x2):
    # Get upper triangle indices
    idx = jnp.triu_indices(n=n, k=1)

    # Compute absolute differences
    x_diff = jnp.abs(x[idx[0]] - x[idx[1]])

    # Compute OR operation for x2
    x2_or = (x2[idx[0]] + x2[idx[1]] == 1).astype(jnp.float32)

    return x_diff, x2_or


def generate_treatments(rng, n, pz=0.5):
    return jnp.array(rng.binomial(n=1, p=pz, size=n), dtype=jnp.float32)


def CAR_cov(triu_vals, sig_inv, rho, n):
    # Cov(Y) = \Sigma = sig_inv * (D - rho*A)^-1
    # So precision = \Sigma^{-1} = (1/sig_inv) * (D - rho*A)
    adj_mat = utils.Triu_to_mat(triu_vals)
    degs_diag = jnp.sum(adj_mat, axis=1) * jnp.eye(n)
    # Compute precision matrix Sigma^{-1}
    precis_ = sig_inv * (degs_diag - rho * adj_mat)
    # Return Sigma
    return jnp.linalg.inv(precis_)


def generate_outcomes_n_exposures(rng, n, x, z, triu_star, eta, rho, sig_inv):
    # Compute exposures
    expos = utils.compute_exposures(triu_star, x)

    # Expectation vector
    df_nodes = jnp.transpose(jnp.stack([jnp.ones(n), z, x, expos]))
    mean_y = df_nodes @ eta

    # Generate observed outcomes
    y = jnp.array(
        rng.multivariate_normal(mean_y, CAR_cov(triu_star, sig_inv, rho, n)),
        dtype=jnp.float32,
    )

    return y, expos


def generate_triu_star(rng, triu_dim, x2_or, theta):
    probs = expit(theta[0] + theta[1] * x2_or)
    triu_star = jnp.array(rng.binomial(n=1, p=probs, size=triu_dim), dtype=jnp.float32)
    return triu_star


def generate_fixed_data(rng, n, param, pz=0.5):
    # covaraites + treatments
    x, x2 = generate_covariates(rng, n)
    x_diff, x2_or = compute_pairwise_diffs(n, x, x2)
    Z = generate_treatments(rng, n, pz)

    # triu_star (A*)
    triu_dim = n * (n - 1) // 2
    triu_star = generate_triu_star(rng, triu_dim, x2_or, param.theta)

    # outcomes + exposures
    Y, true_exposures = generate_outcomes_n_exposures(
        rng, n, x, Z, triu_star, param.eta, param.rho, param.sig_inv
    )

    return {
        "x": x,
        "x2": x2,
        "x_diff": x_diff,
        "x2_or": x2_or,
        "triu_star": triu_star,
        "Z": Z,
        "true_exposures": true_exposures,
        "Y": Y,
    }


def generate_proxy_networks(rng, triu_dim, triu_star, gamma, x_diff, Z):
    # first proxy
    probs_obs = expit(
        triu_star * gamma[0] + (1 - triu_star) * (gamma[1] + gamma[2] * x_diff)
    )
    triu_obs = jnp.array(
        rng.binomial(n=1, p=probs_obs, size=triu_dim), dtype=jnp.float32
    )
    obs_exposures = utils.compute_exposures(triu_obs, Z)

    # repeated proxy
    # TODO: probs_obs_rep =
    # TODO: triu_obs_rep
    # TODO: return triu_obs, triu_obs_rep
    return {"triu_obs": triu_obs, "obs_exposures": obs_exposures}


def data_for_sim(fixed_data_dict, obs_triu_dict):
    return utils.DataTuple(
        x=fixed_data_dict["x"],
        x2=fixed_data_dict["x2"],
        x_diff=fixed_data_dict["x_diff"],
        x2_or=fixed_data_dict["x2_or"],
        triu_star=fixed_data_dict["triu_star"],
        triu_obs=obs_triu_dict["triu_obs"],
        Z=fixed_data_dict["Z"],
        obs_exposures=obs_triu_dict["obs_exposures"],
        true_exposures=fixed_data_dict["true_exposures"],
        Y=fixed_data_dict["Y"],
    )
