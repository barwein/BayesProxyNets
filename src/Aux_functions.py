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
    def __init__(self, n, theta, eta, sig_y):
        self.n = n
        self.theta = theta
        self.eta = eta
        self.sig_y = sig_y
        self.X = self.generate_X()
        self.Z = self.generate_Z()
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


df1 = DataGeneration(n=10,theta=[0.5,0.5],eta=[-1,1.5,0.5,-0.25],sig_y=1).get_data()
df2 = DataGeneration(n=10,theta=[0.5,0.5],eta=[-1,1.5,0.5,-0.25],sig_y=1).get_data()
print(df1)
print(df2)
print("X equal?", np.array_equal(df1["X"], df2["X"]))
print("exposures equal? ", np.array_equal(df1["exposures"], df2["exposures"]))
print("triu equal? ", np.array_equal(df1["triu"], df2["triu"]))









