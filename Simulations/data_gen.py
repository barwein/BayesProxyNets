###
# Functions that generate data for simulations
###

import jax.numpy as jnp
from jax.scipy.special import expit
from jax import random
import numpy as np
import src.utils as utils

# --- Data generation functions ---


def generate_covariates(rng, n):
    keys = random.split(rng, 2)
    x = jnp.astype(random.normal(key=keys[0], shape=(n,)), jnp.float32)
    x2 = jnp.astype(random.bernoulli(key=keys[1], p=0.1, shape=(n,)), jnp.float32)
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
    # return jnp.array(rng.binomial(n=1, p=pz, size=n), dtype=jnp.float32)
    return jnp.astype(random.bernoulli(key=rng, p=pz, shape=(n,)), jnp.float32)


def generate_treatments_prop_to_degree(rng, n, triu_star, p_z=0.5):
    # Probs proportional to degree centrality
    adj_mat = utils.Triu_to_mat(triu_star)
    deg_cen = utils.degree_centrality(adj_mat)
    probs = 6 * p_z * deg_cen
    props = jnp.clip(probs, a_min=0.0, a_max=1.0)
    return jnp.astype(random.bernoulli(key=rng, p=props, shape=(n,)), jnp.float32)


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
    expos = utils.compute_exposures(triu_star, z)

    # Expectation vector
    df_nodes = jnp.transpose(jnp.stack([jnp.ones(n), z, x, expos]))
    mean_y = df_nodes @ eta

    # y_cov = CAR_cov(triu_star, sig_inv, rho, n)
    y_cov = jnp.eye(n) * sig_inv

    y = random.multivariate_normal(key=rng, mean=mean_y, cov=y_cov)

    return y, expos


def generate_triu_star(rng, triu_dim, x2_or, theta):
    probs = expit(theta[0] + theta[1] * x2_or)
    # triu_star = jnp.array(rng.binomial(n=1, p=probs, size=triu_dim), dtype=jnp.float32)
    triu_star = jnp.astype(random.bernoulli(key=rng, p=probs), jnp.float32)
    return triu_star


def generate_fixed_data(rng, n, param, pz=0.5):
    # covaraites + treatments
    keys = random.split(rng, 4)
    x, x2 = generate_covariates(keys[0], n)
    x_diff, x2_or = compute_pairwise_diffs(n, x, x2)

    # triu_star (A*)
    triu_dim = n * (n - 1) // 2
    # triu_star = generate_triu_star(rng, triu_dim, x2_or, param.theta)
    triu_star = generate_triu_star(keys[2], triu_dim, x2_or, param["theta"])

    # treatments
    # Z = generate_treatments(keys[1], n, pz)
    Z = generate_treatments_prop_to_degree(keys[1], n, triu_star, pz)

    # outcomes + exposures
    Y, true_exposures = generate_outcomes_n_exposures(
        keys[3], n, x, Z, triu_star, param["eta"], param["rho"], param["sig_inv"]
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
    keys = random.split(rng, 2)
    probs_obs = expit(
        triu_star * gamma[0] + (1.0 - triu_star) * (gamma[1] + gamma[2] * x_diff)
    )
    # triu_obs = jnp.array(rng.binomial(n=1, p=probs_obs, size=triu_dim), dtype=jnp.float32)
    triu_obs = jnp.astype(random.bernoulli(key=keys[0], p=probs_obs), jnp.float32)
    obs_exposures = utils.compute_exposures(triu_obs, Z)

    logits_obs_rep = triu_star * gamma[0] + (1.0 - triu_star) * (
        gamma[1] + gamma[2] * x_diff
    )
    # logits_obs_rep = triu_star * (gamma[3] + gamma[4] * triu_obs) + (
    #     1.0 - triu_star
    # ) * (gamma[5] + gamma[6] * triu_obs)

    probs_obs_rep = expit(logits_obs_rep)

    triu_obs_rep = jnp.astype(
        random.bernoulli(key=keys[1], p=probs_obs_rep), jnp.float32
    )

    return {
        "triu_obs": triu_obs,
        "obs_exposures": obs_exposures,
        "triu_obs_rep": triu_obs_rep,
    }


def data_for_sim(fixed_data_dict, obs_triu_dict):
    return utils.DataTuple(
        x=fixed_data_dict["x"],
        x2=fixed_data_dict["x2"],
        x_diff=fixed_data_dict["x_diff"],
        x2_or=fixed_data_dict["x2_or"],
        triu_star=fixed_data_dict["triu_star"],
        triu_obs=obs_triu_dict["triu_obs"],
        triu_obs_rep=obs_triu_dict["triu_obs_rep"],
        Z=fixed_data_dict["Z"],
        obs_exposures=obs_triu_dict["obs_exposures"],
        true_exposures=fixed_data_dict["true_exposures"],
        Y=fixed_data_dict["Y"],
    )


def dynamic_intervention(x, thresholds=(0.75, 1.5)):
    # Dynamic intervention by 'x' values
    Z_h1 = jnp.where((x > thresholds[0]) | (x < -thresholds[0]), 1, 0)
    Z_h2 = jnp.where((x > thresholds[1]) | (x < -thresholds[1]), 1, 0)
    return jnp.array([Z_h1, Z_h2], dtype=jnp.float32)


def stochastic_intervention(rng, n, alphas=(0.7, 0.3), n_approx=1000):
    # def stochastic_intervention(rng, n, alphas=(0.7, 0.3), n_approx=50):
    # Stochastic intervention by 'alpha' values
    keys = random.split(rng, 2)
    Z_stoch1 = jnp.astype(
        random.bernoulli(key=keys[0], p=alphas[0], shape=(n_approx, n)), jnp.float32
    )
    Z_stoch2 = jnp.astype(
        random.bernoulli(key=keys[1], p=alphas[1], shape=(n_approx, n)), jnp.float32
    )
    return jnp.array([Z_stoch1, Z_stoch2], dtype=jnp.float32)


def get_true_estimands(n, z_new, triu_star, eta):
    if z_new.ndim == 3:  # stoch intervention
        exposures_new1 = utils.compute_exposures(triu_star, z_new[0, :, :])
        exposures_new2 = utils.compute_exposures(triu_star, z_new[1, :, :])
        exposures_diff = exposures_new1 - exposures_new2
        z_diff = z_new[0, :, :] - z_new[1, :, :]
        n_stoch = z_new.shape[1]
        results = np.zeros((n_stoch, n))
        for i in range(n_stoch):
            results[i, :] = eta[1] * z_diff[i, :] + eta[3] * exposures_diff[i, :]
        return jnp.mean(results, axis=0).squeeze()
    elif z_new.ndim == 2:  # dynamic intervention
        exposures_new1 = utils.compute_exposures(triu_star, z_new[0, :])
        exposures_new2 = utils.compute_exposures(triu_star, z_new[1, :])
        exposures_diff = exposures_new1 - exposures_new2
        z_diff = z_new[0, :] - z_new[1, :]
        results = eta[1] * z_diff + eta[3] * exposures_diff
        return results
    else:
        raise ValueError("Invalid dimension for new interventions")


def new_interventions_estimands(rng, n, x, triu_star, eta):
    # new interventions
    Z_h = dynamic_intervention(x)

    key, _ = random.split(rng)
    Z_stoch = stochastic_intervention(key, n)

    # GATE
    Z_gate = jnp.stack([jnp.ones(n), jnp.zeros(n)])

    # new estimands
    dynamic_estimands = get_true_estimands(n, Z_h, triu_star, eta)
    stoch_estimands = get_true_estimands(n, Z_stoch, triu_star, eta)
    gate_estimands = get_true_estimands(n, Z_gate, triu_star, eta)

    return utils.NewEstimands(
        Z_h=Z_h,
        Z_stoch=Z_stoch,
        Z_gate=Z_gate,
        estimand_h=dynamic_estimands,
        estimand_stoch=stoch_estimands,
        estimand_gate=gate_estimands,
    )
