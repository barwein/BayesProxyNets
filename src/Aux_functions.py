# Load libraries
import time
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc_experimental as pmx
import pytensor as pt
import networkx as nx
import seaborn as sns
import os
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import expit
import numpyro.distributions as dist
import numpyro
from numpyro.contrib.funsor import config_enumerate
from numpyro.contrib.control_flow import scan
from tqdm import tqdm
from joblib import Parallel, delayed
from numpyro.infer import MCMC, HMC, NUTS, DiscreteHMCGibbs, MixedHMC, Predictive

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
RANDOM_SEED = 892357143
rng = np.random.default_rng(RANDOM_SEED)

# Data generation

class DataGeneration:
    def __init__(self, n, theta, eta, sig_y, pz):
        self.n = n
        self.theta = theta
        self.eta = eta
        self.sig_y = sig_y
        self.X = self.generate_X()
        self.Z = self.generate_Z(p=pz)
        self.X_diff = self.x_diff()
        self.triu_dim = int(self.n*(self.n-1)/2)
        self.triu = self.generate_triu()
        self.adj_mat = self.generate_adj_matrix()
        self.exposures = self.generate_exposures()
        self.Y = self.generate_outcome()

    def generate_X(self, loc=0, scale=3):
        return rng.normal(loc=loc,scale=scale,size=self.n)

    def generate_Z(self, p=0.5):
        return rng.binomial(n=1,p=p,size=self.n)

    def x_diff(self):
        x_d = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                x_d.append(np.abs(self.X[i] - self.X[j]))
        return np.array(x_d)

    def generate_triu(self):
        probs = expit(self.theta[0] + self.theta[1] * self.X_diff)
        return rng.binomial(n=1, p=probs, size=self.triu_dim)

    def generate_adj_matrix(self):
        mat = np.zeros((self.n, self.n))
        idx_upper_tri = np.triu_indices(n=self.n, k=1)
        mat[idx_upper_tri] = self.triu
        return mat + mat.T

    def generate_exposures(self):
        return np.dot(self.adj_mat, self.Z)

    def generate_outcome(self):
        mu_y = self.eta[0] + self.eta[1]*self.Z + self.eta[2]*self.exposures + self.eta[3]*self.X
        return mu_y + rng.normal(loc=0,scale=self.sig_y,size=self.n)

    def get_data(self):
        return {"Z" : self.Z, "X" : self.X, "Y" : self.Y,
                "triu" : self.triu, "exposures" : self.exposures,
                "adj_mat" : self.adj_mat, "X_diff" : self.X_diff}


def create_noisy_network(adj_mat, gamma, n):
    obs_mat = np.zeros((n, n))  # create nXn matrix of zeros
    triu_idx = np.triu_indices(n=n, k=1)
    obs_mat[triu_idx] = adj_mat[triu_idx]  # init as true network
    for i in range(0, n):  # add noise
        for j in range(i + 1, n):
            if adj_mat[i, j] == 1:
                obs_mat[i, j] = rng.binomial(n=1, p=1 - gamma[1], size=1)[0]  # retain existing edge w.p. `1-gamma1`
            else:
                obs_mat[i, j] = rng.binomial(n=1, p=gamma[0], size=1)[0]  # add non-existing edge w.p. `gamma0`
    obs_mat = obs_mat + obs_mat.T
    triu_obs = obs_mat[triu_idx]
    return {"obs_mat" : obs_mat,
            "triu_obs" : triu_obs}

def triu_to_mat(TRIU, n):
    AM = np.zeros((n,n))
    AM[np.triu_indices(n=n,k=1)] = TRIU
    return AM + AM.T

@config_enumerate
def network_model(X_diff, TriU, n):
    with numpyro.plate("theta_i", 2):
        theta = numpyro.sample("theta", dist.Normal(0, 10))
    mu_net = theta[0] + theta[1]*X_diff
    triu_n = int(n * (n - 1) / 2)

    gamma0 = numpyro.sample("gamma0", dist.Uniform(low=0, high=0.5))
    gamma1 = numpyro.sample("gamma1", dist.Uniform(low=0, high=0.5))

    with numpyro.plate("A* and A", triu_n):
        triu_star = numpyro.sample("triu_star", dist.Bernoulli(logits=mu_net),
                                   infer={"enumerate": "parallel"})
        prob_misspec = triu_star * (1 - gamma1) + (1 - triu_star) * gamma0
        numpyro.sample("obs_triu", dist.Bernoulli(probs=prob_misspec), obs=TriU)

def outcome_model(Y, Z, X, A, n):
    with numpyro.plate("eta_i", 4):
        eta = numpyro.sample("eta", dist.Normal(0, 10))
    sig = numpyro.sample("sig", dist.Exponential(0.5))
    expos = jnp.dot(A, Z)
    mu_y = eta[0] + eta[1]*Z + eta[2]*expos + eta[3]*X
    with numpyro.plate("n", n):
        numpyro.sample("Y", dist.Normal(loc=mu_y, scale=sig), obs=Y)


class Network_MCMC:
    def __init__(self, data, n, rng_key):
        self.X_diff = data["X_diff"]
        self.triu = data["triu"]
        self.adj_mat = data["adj_mat"]
        self.n = n
        self.rng_key = rng_key
        self.network_m = self.network()
    def network(self):
        kernel = NUTS(network_model)
        return MCMC(kernel, num_warmup=1000, num_samples=3000, num_chains=4, progress_bar=False)
    def run_network_model(self):
        self.network_m.run(self.rng_key, X_diff=self.X_diff, TriU=self.triu, n=self.n)

    def get_network_predictive(self, mean_posterior = False):
        posterior_samples = self.network_m.get_samples()
        if mean_posterior:
            posterior_mean = {"theta": np.expand_dims(np.mean(posterior_samples["theta"], axis=0), -2),
                              "gamma0": np.expand_dims(np.mean(posterior_samples["gamma0"]), -1),
                              "gamma1": np.expand_dims(np.mean(posterior_samples["gamma1"]), -1)}
            return Predictive(model=network_model, posterior_samples=posterior_mean,
                              infer_discrete=True,num_samples=1)
        else:
            posterior_predictive = Predictive(model=network_model, posterior_samples=posterior_samples,
                                              infer_discrete=True)
            return posterior_predictive(self.rng_key, X=self.X_diff, TriU=self.triu, n=self.n)["triu_star"]

class Outcome_MCMC:
    def __init__(self, data, n, rng_key, iter, n_warmup=1000, n_samples=3000, n_chains=4):
        self.X = data["X"]
        self.Z = data["Z"]
        self.Y = data["Y"]
        self.exposures = data["exposures"]
        self.adj_mat = data["adj_mat"]
        self.n = n
        self.rng_key = rng_key
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.outcome_m = self.outcome()
        self.iter = iter

    def outcome(self):
        kernel = NUTS(outcome_model)
        return MCMC(kernel, num_warmup=self.n_warmup, num_samples=self.n_samples,
                    num_chains=self.n_chains, progress_bar=False)
    def run_outcome_model(self):
        self.outcome_m.run(self.rng_key, Y=self.Y,Z=self.Z,X=self.X,A=self.adj_mat,n=self.n)

    def get_summary_outcome_model(self):
        posterior_samples = self.outcome_m.get_samples()
        mean_posterior = np.mean(posterior_samples["eta"],axis=0)[2]
        median_posterior = np.median(posterior_samples["eta"],axis=0)[2]
        std_posterior = np.std(posterior_samples["eta"],axis=0)[2]
        q025_posterior = np.quantile(posterior_samples["eta"],q=0.025,axis=0)[2]
        q975_posterior = np.quantile(posterior_samples["eta"],q=0.975,axis=0)[2]
        return pd.DataFrame({'mean' : mean_posterior,
                             'median' : median_posterior,
                             'std' : std_posterior,
                             'q025' : q025_posterior,
                             'q975' : q975_posterior},
                            index = [self.iter])






if __name__ == "__main__":

    N = 300
    THETA = [-2.5, -0.5]
    GAMMA = [0.05, 0.3]
    ETA = [-1, -3, 0.5, -0.25]
    SIG_Y = 1
    PZ = 0.3

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)


    df_true = DataGeneration(n=N,theta=THETA,eta=ETA,sig_y=SIG_Y,pz=PZ).get_data()
    obs_network = create_noisy_network(df_true["adj_mat"], GAMMA, N)
    df_obs = df_true.copy()
    df_obs["adj_mat"] = obs_network["obs_mat"]
    df_obs["triu"] = obs_network["triu_obs"]

    true_outcome_mcmc = Outcome_MCMC(data=df_true, n=N, rng_key=rng_key, iter = 1)
    true_outcome_mcmc.run_outcome_model()
    true_results = true_outcome_mcmc.get_summary_outcome_model()
    true_results["type"] = "TRUE"
    obs_outcome_mcmc = Outcome_MCMC(data=df_obs, n=N, rng_key=rng_key, iter = 1)
    obs_outcome_mcmc.run_outcome_model()
    obs_results = obs_outcome_mcmc.get_summary_outcome_model()
    obs_results["type"] = "OBSERVED"

    print(pd.concat([true_results, obs_results]))

    # TODO: add class of `cut-posterior` (two and three stages) and `plug-in`
    # TODO: create simulation iterations procedure (in other py file perhaps)



