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
    def __init__(self, n, theta, gamma, eta, sig_y):
        self.n = n
        self.theta = theta
        self.gamma = gamma
        self.eta = eta
        self.sig_y = sig_y
        self.generate_X()
        self.generate_Z()
        self.X_diff()
        self.generate_network()

    def generate_X(self, loc=0, scale=3):
        self.X = rng.normal(loc=loc,scale=scale,size=self.n)

    def generate_Z(self, p=0.5):
        self.Z = rng.binomial(n=1,p=p,size=self.n)

    def X_diff(self):
        x_d = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                x_d.append(np.abs(self.X[i] - self.X[j]))
        self.X_diff = np.array(x_d)

    def generate_network(self):
        probs = expit(self.theta[0] + self.theta[1] * self.X_diff)
        mat = np.zeros((self.n, self.n))
        triu_dim = int(self.n*(self.n-1)/2)
        idx_upper_tri = np.triu_indices(n=self.n, k=1))
        edges = rng.binomial(n=1, p=probs, size=triu_dim)
        mat[idx_upper_tri] = edges
        mat = mat + mat.T
        self.adj_mat = mat
        self.triu = edges

    def generate_exposures(self):
        self.exposures = np.dot(self.adj_mat, self.Z)


