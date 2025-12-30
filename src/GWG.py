###
# Gibbs-With-Gradients sampler for discerte (binary) parameters (A* in our case)
# GWG is a specfic case of Locally Informed Proposals (LIP) samplers for discrete variables
###

import jax.numpy as jnp
from jax import jit
import jax
import jax.random as random
from typing import NamedTuple, Any
from src.Models import (
    triu_star_grad_fn,
    triu_star_grad_fn_rep,
    triu_star_grad_fn_misspec,
    triu_star_grad_fn_rep_misspec,
    triu_star_grad_fn_no_z,
    triu_star_grad_fn_rep_no_z,
)
from src.utils import ParamTuple
from functools import partial  # Required for jit with arguments


# Global hyper-parameters
BATCH_LEN = 1
N_STEPS = 1


# NamedTuple for GWG states and info (IP = Informed Proposals)
class IPState(NamedTuple):
    positions: Any
    logdensity: Any
    logdensity_grad: Any
    scores: Any


class IPInfo(NamedTuple):
    acceptance_rate: Any
    is_accepted: Any


# --- Aux function for edge flip proposals ---
# @jit
@partial(jit, static_argnames=("batch_len",))
def weighted_sample_and_logprobs(key, scores, batch_len=BATCH_LEN):
    """
    Weighted sample with replacement of BATCH_LEN indices using Gumbel-max trick

    Args:
    key: PRNG key
    scores: Scores for the weights of the samples (log density ratios)

    """
    # Get samples using Gumbel-max trick
    gumbel_noise = random.gumbel(key, shape=scores.shape)
    perturbed = scores + gumbel_noise
    # _, selected_indices = jax.lax.top_k(perturbed, BATCH_LEN)
    _, selected_indices = jax.lax.top_k(perturbed, batch_len)
    # Compute log soft-max probs of scores
    log_probs = jax.nn.log_softmax(scores)
    selected_log_probs = log_probs[selected_indices]
    # return indices and log probabilities
    return selected_indices, selected_log_probs.sum()


@jit
def propsal_logprobs(idx, scores):
    """
    Compute log probability using soft-max of scores
    """
    log_probs = jax.nn.log_softmax(scores)
    return log_probs[idx].sum()


# --- GWG samplers (Locally Informed Proposals) ---

### No repeated measures ###


# @jit
@partial(jit, static_argnames=("batch_len",))
def GWG_step(rng_key, state, data, param, batch_len=BATCH_LEN):
    """
    One step of GWG sampler for A* parameter

    Args:
    rng_key: PRNG key
    state: IPState - current state of the sampler
    data: DataTuple - data for the model
    param: ParamTuple - parameters of the model

    Returns:
    IPState: new state of the sampler
    IPInfo: info about the acceptance of the proposal
    """
    key1, key2 = random.split(rng_key, 2)
    # sample idx to propose edge flip
    # idx, forward_logprob = weighted_sample_and_logprobs(key1, state.scores)
    idx, forward_logprob = weighted_sample_and_logprobs(key1, state.scores, batch_len)

    # proposed new state
    new_triu_star = state.positions.at[idx].set(1 - state.positions[idx])

    # backward proposal
    f_proposed, backward_grad = triu_star_grad_fn(new_triu_star, data, param)
    backward_scores = (-(2 * new_triu_star - 1) * backward_grad) / 2
    backward_logprob = propsal_logprobs(idx, backward_scores)

    # accept/reject
    acceptance_ratio = jnp.minimum(
        jnp.exp(f_proposed - state.logdensity + backward_logprob - forward_logprob), 1
    )
    accept = random.uniform(key2) <= acceptance_ratio

    # update state and info
    new_triu_star = jax.lax.select(accept, new_triu_star, state.positions)
    new_logpost = jax.lax.select(accept, f_proposed, state.logdensity)
    new_grad = jax.lax.select(accept, backward_grad, state.logdensity_grad)
    new_scores = jax.lax.select(accept, backward_scores, state.scores)

    state = IPState(new_triu_star, new_logpost, new_grad, new_scores)
    info = IPInfo(acceptance_ratio, accept)

    return state, info


@partial(jit, static_argnames=("batch_len",))
def GWG_step_no_z(rng_key, state, data, param, batch_len=BATCH_LEN):
    """
    One step of GWG sampler for A* parameter

    No treatment model version

    Args:
    rng_key: PRNG key
    state: IPState - current state of the sampler
    data: DataTuple - data for the model
    param: ParamTuple - parameters of the model

    Returns:
    IPState: new state of the sampler
    IPInfo: info about the acceptance of the proposal
    """
    key1, key2 = random.split(rng_key, 2)
    # sample idx to propose edge flip
    idx, forward_logprob = weighted_sample_and_logprobs(key1, state.scores, batch_len)

    # proposed new state
    new_triu_star = state.positions.at[idx].set(1 - state.positions[idx])

    # backward proposal
    f_proposed, backward_grad = triu_star_grad_fn_no_z(new_triu_star, data, param)
    backward_scores = (-(2 * new_triu_star - 1) * backward_grad) / 2
    backward_logprob = propsal_logprobs(idx, backward_scores)

    # accept/reject
    acceptance_ratio = jnp.minimum(
        jnp.exp(f_proposed - state.logdensity + backward_logprob - forward_logprob), 1
    )
    accept = random.uniform(key2) <= acceptance_ratio

    # update state and info
    new_triu_star = jax.lax.select(accept, new_triu_star, state.positions)
    new_logpost = jax.lax.select(accept, f_proposed, state.logdensity)
    new_grad = jax.lax.select(accept, backward_grad, state.logdensity_grad)
    new_scores = jax.lax.select(accept, backward_scores, state.scores)

    state = IPState(new_triu_star, new_logpost, new_grad, new_scores)
    info = IPInfo(acceptance_ratio, accept)

    return state, info


@partial(jit, static_argnames=("batch_len",))
def GWG_step_misspec(rng_key, state, data, param, batch_len=BATCH_LEN):
    """
    One step of GWG sampler for A* parameter

    Misspecified model version (no x and x_diff)

    Args:
    rng_key: PRNG key
    state: IPState - current state of the sampler
    data: DataTuple - data for the model
    param: ParamTuple - parameters of the model

    Returns:
    IPState: new state of the sampler
    IPInfo: info about the acceptance of the proposal
    """
    key1, key2 = random.split(rng_key, 2)
    # sample idx to propose edge flip
    idx, forward_logprob = weighted_sample_and_logprobs(key1, state.scores, batch_len)

    # proposed new state
    new_triu_star = state.positions.at[idx].set(1 - state.positions[idx])

    # backward proposal
    f_proposed, backward_grad = triu_star_grad_fn_misspec(new_triu_star, data, param)
    backward_scores = (-(2 * new_triu_star - 1) * backward_grad) / 2
    backward_logprob = propsal_logprobs(idx, backward_scores)

    # accept/reject
    acceptance_ratio = jnp.minimum(
        jnp.exp(f_proposed - state.logdensity + backward_logprob - forward_logprob), 1
    )
    accept = random.uniform(key2) <= acceptance_ratio

    # update state and info
    new_triu_star = jax.lax.select(accept, new_triu_star, state.positions)
    new_logpost = jax.lax.select(accept, f_proposed, state.logdensity)
    new_grad = jax.lax.select(accept, backward_grad, state.logdensity_grad)
    new_scores = jax.lax.select(accept, backward_scores, state.scores)

    state = IPState(new_triu_star, new_logpost, new_grad, new_scores)
    info = IPInfo(acceptance_ratio, accept)

    return state, info


# @jit
@partial(jit, static_argnames=("n_steps", "batch_len"))
def GWG_kernel(rng_key, state, data, param, n_steps=N_STEPS, batch_len=BATCH_LEN):
    """
    Run N_STEPS of GWG sampler given the current param (continuous) values
    """

    def body_fun(carry, _):
        rng_key, cur_state = carry
        rng_key, step_key = random.split(rng_key)
        # new_state, info = GWG_step(step_key, cur_state, data, param)
        new_state, info = GWG_step(step_key, cur_state, data, param, batch_len)
        return (rng_key, new_state), info

    # Run the scan
    (_, final_state), final_info = jax.lax.scan(
        # body_fun, (rng_key, state), jnp.arange(N_STEPS)
        body_fun,
        (rng_key, state),
        jnp.arange(n_steps),
    )

    return final_state, final_info


@partial(jit, static_argnames=("n_steps", "batch_len"))
def GWG_kernel_no_z(rng_key, state, data, param, n_steps=N_STEPS, batch_len=BATCH_LEN):
    """
    Run N_STEPS of GWG sampler given the current param (continuous) values
    No treatment model version
    """

    def body_fun(carry, _):
        rng_key, cur_state = carry
        rng_key, step_key = random.split(rng_key)
        new_state, info = GWG_step_no_z(
            step_key, cur_state, data, param, batch_len=batch_len
        )
        return (rng_key, new_state), info

    # Run the scan
    (_, final_state), final_info = jax.lax.scan(
        body_fun, (rng_key, state), jnp.arange(n_steps)
    )

    return final_state, final_info


@partial(jit, static_argnames=("n_steps", "batch_len"))
def GWG_kernel_misspec(
    rng_key, state, data, param, n_steps=N_STEPS, batch_len=BATCH_LEN
):
    """
    Run N_STEPS of GWG sampler given the current param (continuous) values

    Misspecified model version (no x and x_diff)
    """

    def body_fun(carry, _):
        rng_key, cur_state = carry
        rng_key, step_key = random.split(rng_key)
        new_state, info = GWG_step_misspec(
            step_key, cur_state, data, param, batch_len=batch_len
        )
        return (rng_key, new_state), info

    # Run the scan
    (_, final_state), final_info = jax.lax.scan(
        body_fun, (rng_key, state), jnp.arange(n_steps)
    )

    return final_state, final_info


# Adapt the GWG sampler for the Metroplis-within-Gibbs (MWG) combined sampler (continuous and discrete parameters)


def make_gwg_gibbs_fn(data, n_steps=N_STEPS, batch_len=BATCH_LEN):
    """
    Will return a function that can be used as a Gibbs step for A* in the MWG sampler;
    The function will be used in the HMCGibbs sampler;
    It will depend on the current continuous parameters (gibbs_sites) and the fixed data
    """

    # Create a closure over the fixed data
    @jit  # Can JIT since data is fixed in closure
    def gwg_gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        # Get current state of triu_star
        cur_triu_star = jnp.astype(gibbs_sites["triu_star"], jnp.float32)

        # Get current values of continuous parameters
        cur_param = ParamTuple(
            theta=hmc_sites["theta"],
            gamma=hmc_sites["gamma"],
            eta=hmc_sites["eta"],
            # rho=hmc_sites["rho"],
            sig_inv=hmc_sites["sig_inv"],
        )

        # Create/update IPState
        cur_logdensity, cur_grad = triu_star_grad_fn(cur_triu_star, data, cur_param)

        cur_scores = (-(2 * cur_triu_star - 1) * cur_grad) / 2

        state = IPState(
            positions=cur_triu_star,
            logdensity=cur_logdensity,
            logdensity_grad=cur_grad,
            scores=cur_scores,
        )

        # Run GWG (N_STEPS times)
        rng_key, _ = random.split(rng_key)
        new_state, _ = GWG_kernel(
            rng_key=rng_key,
            state=state,
            data=data,
            param=cur_param,
            n_steps=n_steps,
            batch_len=batch_len,
        )

        # Return only the new positions as required by HMCGibbs
        # Note that we need to convert the positions to int32 as it is latent binary variable
        # See NumPyro conventions for coding latent variables
        return {"triu_star": jnp.astype(new_state.positions, jnp.int32)}

    return gwg_gibbs_fn


def make_gwg_gibbs_fn_no_z(data, n_steps=N_STEPS, batch_len=BATCH_LEN):
    """
    Will return a function that can be used as a Gibbs step for A* in the MWG sampler;
    The function will be used in the HMCGibbs sampler;
    It will depend on the current continuous parameters (gibbs_sites) and the fixed data
    No treatment model version
    """

    # Create a closure over the fixed data
    @jit  # Can JIT since data is fixed in closure
    def gwg_gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        # Get current state of triu_star
        cur_triu_star = jnp.astype(gibbs_sites["triu_star"], jnp.float32)

        # Get current values of continuous parameters
        cur_param = ParamTuple(
            theta=hmc_sites["theta"],
            gamma=hmc_sites["gamma"],
            eta=hmc_sites["eta"],
            # rho=hmc_sites["rho"],
            sig_inv=hmc_sites["sig_inv"],
        )

        # Create/update IPState
        cur_logdensity, cur_grad = triu_star_grad_fn_no_z(
            cur_triu_star,
            data,
            cur_param,
        )

        cur_scores = (-(2 * cur_triu_star - 1) * cur_grad) / 2

        state = IPState(
            positions=cur_triu_star,
            logdensity=cur_logdensity,
            logdensity_grad=cur_grad,
            scores=cur_scores,
        )

        # Run GWG (N_STEPS times)
        rng_key, _ = random.split(rng_key)
        new_state, _ = GWG_kernel_no_z(
            rng_key=rng_key,
            state=state,
            data=data,
            param=cur_param,
            n_steps=n_steps,
            batch_len=batch_len,
        )

        # Return only the new positions as required by HMCGibbs
        # Note that we need to convert the positions to int32 as it is latent binary variable
        # See NumPyro conventions for coding latent variables
        return {"triu_star": jnp.astype(new_state.positions, jnp.int32)}

    return gwg_gibbs_fn


def make_gwg_gibbs_fn_misspec(data, n_steps=N_STEPS, batch_len=BATCH_LEN):
    """
    Will return a function that can be used as a Gibbs step for A* in the MWG sampler;
    The function will be used in the HMCGibbs sampler;
    It will depend on the current continuous parameters (gibbs_sites) and the fixed data

    Misspecified model version (no x and x_diff)
    """

    # Create a closure over the fixed data
    @jit  # Can JIT since data is fixed in closure
    def gwg_gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        # Get current state of triu_star
        cur_triu_star = jnp.astype(gibbs_sites["triu_star"], jnp.float32)

        # Get current values of continuous parameters
        cur_param = ParamTuple(
            theta=hmc_sites["theta"],
            gamma=hmc_sites["gamma"],
            eta=hmc_sites["eta"],
            # rho=hmc_sites["rho"],
            sig_inv=hmc_sites["sig_inv"],
        )

        # Create/update IPState
        cur_logdensity, cur_grad = triu_star_grad_fn_misspec(
            cur_triu_star,
            data,
            cur_param,
        )

        cur_scores = (-(2 * cur_triu_star - 1) * cur_grad) / 2

        state = IPState(
            positions=cur_triu_star,
            logdensity=cur_logdensity,
            logdensity_grad=cur_grad,
            scores=cur_scores,
        )

        # Run GWG (N_STEPS times)
        rng_key, _ = random.split(rng_key)
        new_state, _ = GWG_kernel_misspec(
            rng_key=rng_key,
            state=state,
            data=data,
            param=cur_param,
            n_steps=n_steps,
            batch_len=batch_len,
        )

        # Return only the new positions as required by HMCGibbs
        # Note that we need to convert the positions to int32 as it is latent binary variable
        # See NumPyro conventions for coding latent variables
        return {"triu_star": jnp.astype(new_state.positions, jnp.int32)}

    return gwg_gibbs_fn


### With repeated measures ###


@partial(jit, static_argnames=("batch_len",))
def GWG_step_rep(rng_key, state, data, param, batch_len=BATCH_LEN):
    """
    One step of GWG sampler for A* parameter

    with repeated proxy networks

    Args:
    rng_key: PRNG key
    state: IPState - current state of the sampler
    data: DataTuple - data for the model
    param: ParamTuple - parameters of the model

    Returns:
    IPState: new state of the sampler
    IPInfo: info about the acceptance of the proposal
    """
    key1, key2 = random.split(rng_key, 2)
    # sample idx to propose edge flip
    idx, forward_logprob = weighted_sample_and_logprobs(
        key1, state.scores, batch_len=batch_len
    )

    # proposed new state
    new_triu_star = state.positions.at[idx].set(1 - state.positions[idx])

    # backward proposal
    f_proposed, backward_grad = triu_star_grad_fn_rep(new_triu_star, data, param)
    backward_scores = (-(2 * new_triu_star - 1) * backward_grad) / 2
    backward_logprob = propsal_logprobs(idx, backward_scores)

    # accept/reject
    acceptance_ratio = jnp.minimum(
        jnp.exp(f_proposed - state.logdensity + backward_logprob - forward_logprob), 1
    )
    accept = random.uniform(key2) <= acceptance_ratio

    # update state and info
    new_triu_star = jax.lax.select(accept, new_triu_star, state.positions)
    new_logpost = jax.lax.select(accept, f_proposed, state.logdensity)
    new_grad = jax.lax.select(accept, backward_grad, state.logdensity_grad)
    new_scores = jax.lax.select(accept, backward_scores, state.scores)

    state = IPState(new_triu_star, new_logpost, new_grad, new_scores)
    info = IPInfo(acceptance_ratio, accept)

    return state, info


@partial(jit, static_argnames=("n_steps", "batch_len"))
def GWG_kernel_rep(rng_key, state, data, param, n_steps=N_STEPS, batch_len=BATCH_LEN):
    """
    Run N_STEPS of GWG sampler given the current param (continuous) values

    for repeated proxy networks model
    """

    def body_fun(carry, _):
        rng_key, cur_state = carry
        rng_key, step_key = random.split(rng_key)
        new_state, info = GWG_step_rep(
            step_key, cur_state, data, param, batch_len=batch_len
        )
        return (rng_key, new_state), info

    # Run the scan
    (_, final_state), final_info = jax.lax.scan(
        body_fun, (rng_key, state), jnp.arange(n_steps)
    )

    return final_state, final_info


@partial(jit, static_argnames=("batch_len",))
def GWG_step_rep_no_z(rng_key, state, data, param, batch_len=BATCH_LEN):
    """
    One step of GWG sampler for A* parameter

    with repeated proxy networks

    No treatment model version

    Args:
    rng_key: PRNG key
    state: IPState - current state of the sampler
    data: DataTuple - data for the model
    param: ParamTuple - parameters of the model

    Returns:
    IPState: new state of the sampler
    IPInfo: info about the acceptance of the proposal
    """
    key1, key2 = random.split(rng_key, 2)
    # sample idx to propose edge flip
    idx, forward_logprob = weighted_sample_and_logprobs(
        key1, state.scores, batch_len=batch_len
    )

    # proposed new state
    new_triu_star = state.positions.at[idx].set(1 - state.positions[idx])

    # backward proposal
    f_proposed, backward_grad = triu_star_grad_fn_rep_no_z(new_triu_star, data, param)
    backward_scores = (-(2 * new_triu_star - 1) * backward_grad) / 2
    backward_logprob = propsal_logprobs(idx, backward_scores)

    # accept/reject
    acceptance_ratio = jnp.minimum(
        jnp.exp(f_proposed - state.logdensity + backward_logprob - forward_logprob), 1
    )
    accept = random.uniform(key2) <= acceptance_ratio

    # update state and info
    new_triu_star = jax.lax.select(accept, new_triu_star, state.positions)
    new_logpost = jax.lax.select(accept, f_proposed, state.logdensity)
    new_grad = jax.lax.select(accept, backward_grad, state.logdensity_grad)
    new_scores = jax.lax.select(accept, backward_scores, state.scores)

    state = IPState(new_triu_star, new_logpost, new_grad, new_scores)
    info = IPInfo(acceptance_ratio, accept)

    return state, info


@partial(jit, static_argnames=("n_steps", "batch_len"))
def GWG_kernel_rep_no_z(
    rng_key, state, data, param, n_steps=N_STEPS, batch_len=BATCH_LEN
):
    """
    Run N_STEPS of GWG sampler given the current param (continuous) values

    for repeated proxy networks model

    No treatment model version
    """

    def body_fun(carry, _):
        rng_key, cur_state = carry
        rng_key, step_key = random.split(rng_key)
        new_state, info = GWG_step_rep_no_z(
            step_key, cur_state, data, param, batch_len=batch_len
        )
        return (rng_key, new_state), info

    # Run the scan
    (_, final_state), final_info = jax.lax.scan(
        body_fun, (rng_key, state), jnp.arange(n_steps)
    )

    return final_state, final_info


@partial(jit, static_argnames=("batch_len",))
def GWG_step_rep_misspec(rng_key, state, data, param, batch_len=BATCH_LEN):
    """
    One step of GWG sampler for A* parameter

    with repeated proxy networks

    Misspecified model version (no x and x_diff)

    Args:
    rng_key: PRNG key
    state: IPState - current state of the sampler
    data: DataTuple - data for the model
    param: ParamTuple - parameters of the model

    Returns:
    IPState: new state of the sampler
    IPInfo: info about the acceptance of the proposal
    """
    key1, key2 = random.split(rng_key, 2)
    # sample idx to propose edge flip
    idx, forward_logprob = weighted_sample_and_logprobs(
        key1, state.scores, batch_len=batch_len
    )

    # proposed new state
    new_triu_star = state.positions.at[idx].set(1 - state.positions[idx])

    # backward proposal
    f_proposed, backward_grad = triu_star_grad_fn_rep_misspec(
        new_triu_star, data, param
    )
    backward_scores = (-(2 * new_triu_star - 1) * backward_grad) / 2
    backward_logprob = propsal_logprobs(idx, backward_scores)

    # accept/reject
    acceptance_ratio = jnp.minimum(
        jnp.exp(f_proposed - state.logdensity + backward_logprob - forward_logprob), 1
    )
    accept = random.uniform(key2) <= acceptance_ratio

    # update state and info
    new_triu_star = jax.lax.select(accept, new_triu_star, state.positions)
    new_logpost = jax.lax.select(accept, f_proposed, state.logdensity)
    new_grad = jax.lax.select(accept, backward_grad, state.logdensity_grad)
    new_scores = jax.lax.select(accept, backward_scores, state.scores)

    state = IPState(new_triu_star, new_logpost, new_grad, new_scores)
    info = IPInfo(acceptance_ratio, accept)

    return state, info


@partial(jit, static_argnames=("n_steps", "batch_len"))
def GWG_kernel_rep_misspec(
    rng_key, state, data, param, n_steps=N_STEPS, batch_len=BATCH_LEN
):
    """
    Run N_STEPS of GWG sampler given the current param (continuous) values

    for repeated proxy networks model

    Misspecified model version (no x and x_diff)
    """

    def body_fun(carry, _):
        rng_key, cur_state = carry
        rng_key, step_key = random.split(rng_key)
        new_state, info = GWG_step_rep_misspec(
            step_key, cur_state, data, param, batch_len=batch_len
        )
        return (rng_key, new_state), info

    # Run the scan
    (_, final_state), final_info = jax.lax.scan(
        body_fun, (rng_key, state), jnp.arange(n_steps)
    )

    return final_state, final_info


# Adapt the GWG sampler for the Metroplis-within-Gibbs (MWG) combined sampler (continuous and discrete parameters)


def make_gwg_gibbs_fn_rep(data, n_steps=N_STEPS, batch_len=BATCH_LEN):
    """
    Will return a function that can be used as a Gibbs step for A* in the MWG sampler;
    Repeated measures model;
    The function will be used in the HMCGibbs sampler;
    It will depend on the current continuous parameters (gibbs_sites) and the fixed data
    """

    # Create a closure over the fixed data
    @jit  # Can JIT since data is fixed in closure
    def gwg_gibbs_fn_rep(rng_key, gibbs_sites, hmc_sites):
        # Get current state of triu_star
        cur_triu_star = jnp.astype(gibbs_sites["triu_star"], jnp.float32)

        # Get current values of continuous parameters
        cur_param = ParamTuple(
            theta=hmc_sites["theta"],
            gamma=hmc_sites["gamma"],
            eta=hmc_sites["eta"],
            # rho=hmc_sites["rho"],
            sig_inv=hmc_sites["sig_inv"],
        )

        # Create/update IPState
        cur_logdensity, cur_grad = triu_star_grad_fn_rep(cur_triu_star, data, cur_param)

        cur_scores = (-(2 * cur_triu_star - 1) * cur_grad) / 2

        state = IPState(
            positions=cur_triu_star,
            logdensity=cur_logdensity,
            logdensity_grad=cur_grad,
            scores=cur_scores,
        )

        # Run GWG (N_STEPS times)
        rng_key, _ = random.split(rng_key)
        new_state, _ = GWG_kernel_rep(
            rng_key=rng_key,
            state=state,
            data=data,
            param=cur_param,
            n_steps=n_steps,
            batch_len=batch_len,
        )

        # Return only the new positions as required by HMCGibbs
        # Note that we need to convert the positions to int32 as it is latent binary variable
        # See NumPyro conventions for coding latent variables
        return {"triu_star": jnp.astype(new_state.positions, jnp.int32)}

    return gwg_gibbs_fn_rep


def make_gwg_gibbs_fn_rep_no_z(data, n_steps=N_STEPS, batch_len=BATCH_LEN):
    """
    Will return a function that can be used as a Gibbs step for A* in the MWG sampler;
    Repeated measures model;
    The function will be used in the HMCGibbs sampler;
    It will depend on the current continuous parameters (gibbs_sites) and the fixed data

    No treatment model version
    """

    # Create a closure over the fixed data
    @jit  # Can JIT since data is fixed in closure
    def gwg_gibbs_fn_rep(rng_key, gibbs_sites, hmc_sites):
        # Get current state of triu_star
        cur_triu_star = jnp.astype(gibbs_sites["triu_star"], jnp.float32)

        # Get current values of continuous parameters
        cur_param = ParamTuple(
            theta=hmc_sites["theta"],
            gamma=hmc_sites["gamma"],
            eta=hmc_sites["eta"],
            # rho=hmc_sites["rho"],
            sig_inv=hmc_sites["sig_inv"],
        )

        # Create/update IPState
        cur_logdensity, cur_grad = triu_star_grad_fn_rep_no_z(
            cur_triu_star, data, cur_param
        )

        cur_scores = (-(2 * cur_triu_star - 1) * cur_grad) / 2

        state = IPState(
            positions=cur_triu_star,
            logdensity=cur_logdensity,
            logdensity_grad=cur_grad,
            scores=cur_scores,
        )

        # Run GWG (N_STEPS times)
        rng_key, _ = random.split(rng_key)
        new_state, _ = GWG_kernel_rep_no_z(
            rng_key=rng_key,
            state=state,
            data=data,
            param=cur_param,
            n_steps=n_steps,
            batch_len=batch_len,
        )

        # Return only the new positions as required by HMCGibbs
        # Note that we need to convert the positions to int32 as it is latent binary variable
        # See NumPyro conventions for coding latent variables
        return {"triu_star": jnp.astype(new_state.positions, jnp.int32)}

    return gwg_gibbs_fn_rep


def make_gwg_gibbs_fn_rep_misspec(data, n_steps=N_STEPS, batch_len=BATCH_LEN):
    """
    Will return a function that can be used as a Gibbs step for A* in the MWG sampler;
    Repeated measures model;
    The function will be used in the HMCGibbs sampler;
    It will depend on the current continuous parameters (gibbs_sites) and the fixed data

    Misspecified model version (no x and x_diff)
    """

    # Create a closure over the fixed data
    @jit  # Can JIT since data is fixed in closure
    def gwg_gibbs_fn_rep(rng_key, gibbs_sites, hmc_sites):
        # Get current state of triu_star
        cur_triu_star = jnp.astype(gibbs_sites["triu_star"], jnp.float32)

        # Get current values of continuous parameters
        cur_param = ParamTuple(
            theta=hmc_sites["theta"],
            gamma=hmc_sites["gamma"],
            eta=hmc_sites["eta"],
            # rho=hmc_sites["rho"],
            sig_inv=hmc_sites["sig_inv"],
        )

        # Create/update IPState
        cur_logdensity, cur_grad = triu_star_grad_fn_rep_misspec(
            cur_triu_star, data, cur_param
        )

        cur_scores = (-(2 * cur_triu_star - 1) * cur_grad) / 2

        state = IPState(
            positions=cur_triu_star,
            logdensity=cur_logdensity,
            logdensity_grad=cur_grad,
            scores=cur_scores,
        )

        # Run GWG (N_STEPS times)
        rng_key, _ = random.split(rng_key)
        new_state, _ = GWG_kernel_rep_misspec(
            rng_key=rng_key,
            state=state,
            data=data,
            param=cur_param,
            n_steps=n_steps,
            batch_len=batch_len,
        )

        # Return only the new positions as required by HMCGibbs
        # Note that we need to convert the positions to int32 as it is latent binary variable
        # See NumPyro conventions for coding latent variables
        return {"triu_star": jnp.astype(new_state.positions, jnp.int32)}

    return gwg_gibbs_fn_rep
