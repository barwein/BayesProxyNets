import jax
import jax.numpy as jnp


def rbf_kernel_matrix(X: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """
    Compute the pairwise RBF kernel matrix for a set of samples.

    Args:
        X: Array of shape (M, d) containing M samples of a continuous parameter.
        sigma: Bandwidth parameter of the RBF kernel.

    Returns:
        A kernel matrix of shape (M, M) with entries k(x_i, x_j).
    """
    # Compute squared Euclidean distances between all pairs of samples.
    sq_dists = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-sq_dists / (2 * sigma**2))


def rbf_kernel_vector(X: jnp.ndarray, y: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """
    Compute the RBF kernel values between each sample in X and a single vector y.

    Args:
        X: Array of shape (M, d) containing M samples of a continuous parameter.
        y: Array of shape (d,) representing the true value.
        sigma: Bandwidth parameter of the RBF kernel.

    Returns:
        A vector of shape (M,) where the i-th entry is k(x_i, y).
    """
    sq_dists = jnp.sum((X - y) ** 2, axis=-1)
    return jnp.exp(-sq_dists / (2 * sigma**2))


def delta_kernel_matrix(X: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the pairwise delta kernel matrix for binary (or discrete) data.
    The delta kernel returns 1 if two vectors are exactly equal, and 0 otherwise.

    Args:
        X: Array of shape (M, d) containing M samples of a binary variable.

    Returns:
        A kernel matrix of shape (M, M) with entries 1 if the samples are equal, else 0.
    """
    eq = jnp.all(X[:, None, :] == X[None, :, :], axis=-1)
    return eq.astype(jnp.float32)


def delta_kernel_vector(X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the delta kernel values between each sample in X and a true binary vector y.

    Args:
        X: Array of shape (M, d) containing M samples.
        y: Array of shape (d,) representing the true binary value.

    Returns:
        A vector of shape (M,) where the i-th entry is 1 if x_i equals y, else 0.
    """
    eq = jnp.all(X == y, axis=-1)
    return eq.astype(jnp.float32)


@jax.jit
def compute_mmd(
    posterior_samples: dict, true_values: dict, rbf_sigma: float = 1.0
) -> jnp.ndarray:
    """
    Computes the Maximum Mean Discrepancy (MMD) between the joint posterior samples and
    the true parameter values. The joint kernel is defined as the product over individual
    kernels for each latent variable.

    For continuous parameters (e.g., 'eta', 'gamma', 'theta'), an RBF kernel is used:
      k(x,y) = exp(-||x-y||^2/(2*sigma^2)).
    For binary parameters (here, 'A_star'), a delta kernel is used:
      k(x,y) = 1 if x equals y, else 0.

    The MMD^2 is computed via the biased estimator
      MMD^2 = (1/M^2) sum_{i,j} k(x_i, x_j) - (2/M) sum_i k(x_i, y) + k(y,y),
    where the true value y is assumed to be a single point (degenerate distribution), so k(y,y)=1.

    Args:
        posterior_samples: Dictionary containing M posterior samples for each parameter.
            Each item is an array of shape (M, k_i), where k_i is the dimensionality of the parameter.
        true_values: Dictionary containing the true value for each parameter.
            Each item is an array of shape (k_i,).
        rbf_sigma: Bandwidth parameter for the RBF kernel (used for continuous parameters).

    Returns:
        mmd2: A scalar (jax.numpy.ndarray) representing the MMD^2 between the posterior samples
              and the true parameter values.
    """
    # Initialize the joint kernel matrix (for samples-samples) and vector (for samples-true)
    joint_K = None  # Will have shape (M, M)
    joint_k_vec = None  # Will have shape (M,)

    # Loop over each latent parameter.
    for key in posterior_samples:
        X = posterior_samples[key]  # Shape: (M, k_i)
        y = true_values[key]  # Shape: (k_i,)

        if key == "triu_star":
            # Use delta kernel for binary latent variable.
            K_param = delta_kernel_matrix(X)
            k_param = delta_kernel_vector(X, y)
        else:
            # Use RBF kernel for continuous parameters.
            K_param = rbf_kernel_matrix(X, rbf_sigma)
            k_param = rbf_kernel_vector(X, y, rbf_sigma)

        # Multiply the kernel contributions from each parameter (i.e. assume product kernel).
        if joint_K is None:
            joint_K = K_param
            joint_k_vec = k_param
        else:
            joint_K = joint_K * K_param
            joint_k_vec = joint_k_vec * k_param

    # Compute the biased MMD^2 estimator.
    mmd2 = jnp.mean(joint_K) - 2.0 * jnp.mean(joint_k_vec) + 1.0
    return mmd2
