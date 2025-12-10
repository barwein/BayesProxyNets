import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import ClippedAdam
import jax.numpy as jnp
from jax import random, vmap
import jax
from jax.scipy.special import logit

import src.Models as models
import src.utils as utils


# --- continuous relaxation of network ---


class ContinuousRelaxationSampler:
    def __init__(
        self,
        rng_key,
        data,
        num_steps=20000,  # SVI steps (optimization iterations)
        learning_rate=0.001,  # Learning rate for Adam
        num_samples=5000,  # Number of samples to draw from the guided posterior
        temperature=0.5,  # Temp for RelaxedBernoulli
        progress_bar=True,
    ):
        """
        SVI-based Sampler for the RelaxedBernoulli baseline.
        Approximates posterior using a Multivariate Normal guide and optimizes ELBO.
        """
        self.rng_key = rng_key
        self.data = data
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.num_samples = num_samples
        self.temperature = temperature
        self.progress_bar = progress_bar

        # 1. Define Guide (Variational Distribution)
        # It handles constraints (like (0,1) for triu_star_soft) automatically.
        self.guide = AutoNormal(models.continuous_relaxation_model)

        # 2. Define Optimizer
        self.optimizer = ClippedAdam(step_size=learning_rate)

        # 3. Define SVI Object
        self.svi = SVI(
            model=models.continuous_relaxation_model,
            guide=self.guide,
            optim=self.optimizer,
            loss=Trace_ELBO(),
        )

        self.svi_result = None
        self.posterior_samples = None

        self.pred_func = None

    def run(self):
        """Run SVI optimization"""
        print(
            f"Running SVI for Continuous Relaxation (Temp={self.temperature}, Steps={self.num_steps})..."
        )

        # Run optimization
        self.svi_result = self.svi.run(
            self.rng_key,
            self.num_steps,
            self.data,
            temperature=self.temperature,  # Pass temp to model
            progress_bar=self.progress_bar,
        )

        # Sample from the guide (Posterior Approximation)
        sample_key, _ = random.split(self.rng_key)

        # Predictive helper to sample from guide
        # This draws samples of the latent variables (params + A*)
        self.pred_func = Predictive(
            model=models.continuous_relaxation_model,
            guide=self.guide,
            params=self.svi_result.params,
            num_samples=self.num_samples,
        )

        self.posterior_samples = self.pred_func(
            sample_key, self.data, temperature=self.temperature
        )
        # return self.posterior_samples

    def get_samples(self):
        if self.posterior_samples is None:
            raise ValueError("Run SVI first using .run()")
        return self.posterior_samples

    def sample_pred_y(self, new_z):
        if self.posterior_samples is None:
            self.run()

        # pred_obs = Predictive(
        #     continuous_relaxation_model,
        #     self.posterior_samples,
        #     return_sites=["Y"]
        # )
        def get_preds(z_input, rng_key):
            d_new = utils.get_data_new_z(z_input, self.data)
            # Pass temperature and rng_key
            # return self.pred_func(rng_key, d_new, temperature=self.temperature)["Y"]
            return self.pred_func(rng_key, d_new, temperature=self.temperature)["Y"]

        if new_z.ndim == 2:  # dynamic treatments (2, N)
            # vmap over the 2 intervention arms
            keys = random.split(random.PRNGKey(1), 2)

            # vmap inputs: z_flat=(2, N), rng_key=(2,)
            preds_both = vmap(get_preds, in_axes=(0, 0))(new_z, keys)

            # preds_both shape: (2, num_samples, N)
            preds = preds_both[0] - preds_both[1]

        elif new_z.ndim == 3:  # stochastic treatments (2, n_approx, N)
            n_approx = new_z.shape[1]
            # keys shape: (2, n_approx, ...)
            keys = random.split(random.PRNGKey(1), 2 * n_approx).reshape(
                2, n_approx, -1
            )

            # Nested vmap:
            # Outer vmap over the 2 intervention arms (axis 0)
            # Inner vmap over the n_approx samples (axis 1 of input)
            vmap_func = vmap(vmap(get_preds, in_axes=(0, 0)), in_axes=(0, 0))

            preds_both = vmap_func(new_z, keys)
            # preds_both shape: (2, n_approx, num_samples, N)

            preds_diff = preds_both[0] - preds_both[1]
            preds = preds_diff.mean(axis=0)

        else:
            raise ValueError("new_z shape must be (2, N) or (2, n_approx, N)")

        return preds

    def new_intervention_error_stats(self, new_z, true_estimands, true_vals):
        """
        Compute error metrics.
        """
        # 1. Wasserstein distance calculation
        # The posterior 'triu_star_soft' contains values in (0,1).
        # post_for_dist = {
        #     "eta": self.posterior_samples["eta"],
        #     "sig_inv": self.posterior_samples["sig_inv"][:, None],
        #     "triu_star": self.posterior_samples["triu_star_soft"]
        # }

        # wasser_dist = utils.compute_1w_distance(post_for_dist, true_vals)
        wasser_dist = 999.0

        # 2. Predict Outcomes
        post_y_preds = self.sample_pred_y(new_z)

        # 3. Compute Stats
        return utils.compute_error_stats(post_y_preds, true_estimands, wasser_dist)
