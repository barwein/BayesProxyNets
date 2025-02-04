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
    expos = utils.compute_exposures(triu_star, z)

    # Expectation vector
    df_nodes = jnp.transpose(jnp.stack([jnp.ones(n), z, x, expos]))
    mean_y = df_nodes @ eta

    y_cov = CAR_cov(triu_star, sig_inv, rho, n)

    # Generate observed outcomes
    y = jnp.array(
        # rng.multivariate_normal(mean_y, CAR_cov(triu_star, sig_inv, rho, n)),
        rng.multivariate_normal(mean_y, y_cov),
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
    # triu_star = generate_triu_star(rng, triu_dim, x2_or, param.theta)
    triu_star = generate_triu_star(rng, triu_dim, x2_or, param["theta"])

    # outcomes + exposures
    Y, true_exposures = generate_outcomes_n_exposures(
        rng, n, x, Z, triu_star, param["eta"], param["rho"], param["rho"]
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


def dynamic_intervention(x, thresholds=(0.75, 1.5)):
    # Dynamic intervention by 'x' values
    Z_h1 = jnp.where((x > thresholds[0]) | (x < -thresholds[0]), 1, 0)
    Z_h2 = jnp.where((x > thresholds[1]) | (x < -thresholds[1]), 1, 0)
    return jnp.array([Z_h1, Z_h2], dtype=jnp.float32)


# def stochastic_intervention(rng, n, alphas=(0.7, 0.3), n_approx=1000):
def stochastic_intervention(rng, n, alphas=(0.7, 0.3), n_approx=50):
    # Stochastic intervention by 'alpha' values
    Z_stoch1 = rng.binomial(n=1, p=alphas[0], size=(n_approx, n))
    Z_stoch2 = rng.binomial(n=1, p=alphas[1], size=(n_approx, n))
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
    Z_stoch = stochastic_intervention(rng, n)

    # new estimands
    dynamic_estimands = get_true_estimands(n, Z_h, triu_star, eta)
    stoch_estimands = get_true_estimands(n, Z_stoch, triu_star, eta)

    return utils.NewEstimands(
        Z_h=Z_h,
        Z_stoch=Z_stoch,
        estimand_h=dynamic_estimands,
        estimand_stoch=stoch_estimands,
    )
