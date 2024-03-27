# Load libraries
import time
import numpy as np
import pandas as pd
import multiprocessing
import os
# import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import expit
import numpyro.distributions as dist
import numpyro
from numpyro.contrib.funsor import config_enumerate
# from tqdm import tqdm
from joblib import Parallel, delayed
from numpyro.infer import MCMC, NUTS, Predictive

N_CORES = multiprocessing.cpu_count()
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
RANDOM_SEED = 892357143
rng = np.random.default_rng(RANDOM_SEED)

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

def q025(x):
    return x.quantile(.025)

def q975(x):
    return x.quantile(.975)

def between_var(x, mean_all):
    n_rep = len(x)
    return (1/(n_rep - 1))*np.sum(np.square(x-mean_all))

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

def outcome_stat_model(Y, Z, X, expos, n):
    with numpyro.plate("eta_i",4):
        eta = numpyro.sample("eta",dist.Normal(0,10))
    sig = numpyro.sample("sig",dist.Exponential(0.5))
    mu_y = eta[0] + eta[1]*Z + eta[2]*expos + eta[3]*X
    with numpyro.plate("n", n):
        numpyro.sample("Y", dist.Normal(loc=mu_y, scale=sig), obs=Y)


class Network_MCMC:
    def __init__(self, data, n, rng_key, n_warmup=1000, n_samples=3000, n_chains=4):
        self.X_diff = data["X_diff"]
        self.triu = data["triu"]
        self.adj_mat = data["adj_mat"]
        self.n = n
        self.rng_key = rng_key
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.network_m = self.network()
    def network(self):
        kernel = NUTS(network_model)
        return MCMC(kernel, num_warmup=self.n_warmup, num_samples=self.n_samples,
                    num_chains=self.n_chains, progress_bar=False)
    def run_network_model(self):
        self.network_m.run(self.rng_key, X_diff=self.X_diff, TriU=self.triu, n=self.n)

    def get_network_predictive(self, mean_posterior = False):
        posterior_samples = self.network_m.get_samples()
        if mean_posterior:
            posterior_mean = {"theta": np.expand_dims(np.mean(posterior_samples["theta"], axis=0), -2),
                              "gamma0": np.expand_dims(np.mean(posterior_samples["gamma0"]), -1),
                              "gamma1": np.expand_dims(np.mean(posterior_samples["gamma1"]), -1)}
            return Predictive(model=network_model, posterior_samples=posterior_mean,
                              infer_discrete=True, num_samples=1)
        else:
            posterior_predictive = Predictive(model=network_model, posterior_samples=posterior_samples,
                                              infer_discrete=True)
            return posterior_predictive(self.rng_key, X_diff=self.X_diff, TriU=self.triu, n=self.n)["triu_star"]

class Outcome_MCMC:
    def __init__(self, data, n, type, rng_key, iter, suff_stat = False, n_warmup=1000, n_samples=3000, n_chains=4):
        self.X = data["X"]
        self.Z = data["Z"]
        self.Y = data["Y"]
        self.exposures = data["exposures"]
        self.adj_mat = data["adj_mat"]
        self.n = n
        self.type = type
        self.rng_key = rng_key
        self.iter = iter
        self.suff_stat = suff_stat
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.outcome_m = self.outcome()

    def outcome(self):
        if not self.suff_stat:
            kernel = NUTS(outcome_model)
        else:
            kernel = NUTS(outcome_stat_model)

        return MCMC(kernel, num_warmup=self.n_warmup, num_samples=self.n_samples,
                    num_chains=self.n_chains, progress_bar=False)
    def run_outcome_model(self):
        if not self.suff_stat:
            self.outcome_m.run(self.rng_key, Y=self.Y,Z=self.Z,X=self.X,A=self.adj_mat,n=self.n)
        else:
            self.outcome_m.run(self.rng_key, Y=self.Y,Z=self.Z,X=self.X,expos=self.exposures,n=self.n)

    def get_summary_outcome_model(self):
        """

        :rtype: object
        """
        posterior_samples = self.outcome_m.get_samples()
        mean_posterior = np.mean(posterior_samples["eta"],axis=0)[2]
        median_posterior = np.median(posterior_samples["eta"],axis=0)[2]
        std_posterior = np.std(posterior_samples["eta"],axis=0)[2]
        q025_posterior = np.quantile(posterior_samples["eta"],q=0.025,axis=0)[2]
        q975_posterior = np.quantile(posterior_samples["eta"],q=0.975,axis=0)[2]
        min_posterior = np.min(posterior_samples["eta"],axis=0)[2]
        max_posterior = np.max(posterior_samples["eta"],axis=0)[2]
        return pd.DataFrame({'mean' : mean_posterior,
                             'median' : median_posterior,
                             'std' : std_posterior,
                             'q025' : q025_posterior,
                             'q975' : q975_posterior,
                             'min' : min_posterior,
                             'max' : max_posterior,
                             'type' : self.type},
                            index = [self.iter])

class Bayes_Modular:
    def __init__(self, data, n, bm_type, post_predictive, n_rep = 1000, n_warmup = 250,
                 n_samples = 500, iter = 0, n_cores = N_CORES):
        self.X = data["X"]
        self.Z = data["Z"]
        self.Y = data["Y"]
        self.exposures = data["exposures"]
        self.X_diff = data["X_diff"]
        self.triu = data["triu"]
        self.adj_mat = data['adj_mat']
        self.n = n
        self.type = bm_type # one of ["cut-2S", "cut-3S", "plugin"]
        self.post_predictive = post_predictive
        self.n_rep = n_rep
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.iter = iter
        self.n_cores = n_cores
        self.Y_mcmc = self.MCMC_obj()
        self.results = None
    def MCMC_obj(self):
        return MCMC(NUTS(outcome_model), num_warmup=self.n_warmup, num_samples=self.n_samples,
                    num_chains=1, progress_bar=False)

    def stage_aux(self, i):
        # sample network
        if self.type == "cut-2S":
            curr_mat = self.post_predictive(random.PRNGKey(i**2),X_diff=self.X_diff,TriU=self.triu,n=self.n)
            curr_mat = triu_to_mat(curr_mat["triu_star"], self.n)
        elif self.type == "cut-3S":
            curr_mat = triu_to_mat(self.post_predictive[i,], self.n)
        else:
            raise TypeError
        # Run MCMC
        self.Y_mcmc.run(random.PRNGKey(i**2), Y=self.Y, Z=self.Z, X=self.X, A=curr_mat, n=self.n)
        curr_posterior_samples = self.Y_mcmc.get_samples()
        # save results
        alpha_shape = curr_posterior_samples["eta"].shape
        converted_post_samp = {"iter": i, "sig": curr_posterior_samples["sig"]}
        for j in range(alpha_shape[1]):
            converted_post_samp["eta" + '_' + str(j)] = curr_posterior_samples["eta"][:, j]
        return pd.DataFrame(converted_post_samp)

    def twostage_inferene(self):
        twostage_post_samp = Parallel(n_jobs=self.n_cores)(delayed(self.stage_aux)(i) for i in range(self.n_rep))
        return pd.concat(twostage_post_samp, axis=0)

    def threestage_inference(self):
        i_range = np.random.choice(a = range(self.post_predictive.shape[0]),
                                   size = self.n_rep, replace = False)
        threestage_post_samp = Parallel(n_jobs=self.n_cores)(delayed(self.stage_aux)(i) for i in i_range)
        return pd.concat(threestage_post_samp, axis=0)

    def plugin_aux(self, i_range):
        # deg_list = []
        expos_list = []
        for i in i_range:
            # sample network
            curr_mat = triu_to_mat(self.post_predictive[i,], self.n)
            # save statistics
            # deg_list.append([np.sum(curr_mat, 1)])
            expos_list.append([np.dot(curr_mat, self.Z)])
        return {'expos': expos_list}
        # return {'deg': deg_list, 'expos': expos_list}

    def plugin_inference(self):
        return self.plugin_aux(range(self.post_predictive.shape[0]))

    def run_inference(self):
        if self.type == "cut-3S":
            self.results = self.threestage_inference()
        if self.type == "cut-2S":
            self.results = self.twostage_inferene()
        if self.type == "plugin":
            self.results = self.plugin_inference()

    def cut_posterior_summarized(self):
        agg_results = self.results.agg(['mean','median',q025, q975,'min','max'])
        mean_eta2 = agg_results["eta_2"]["mean"]
        eta2_agg_by_iter = agg_results[["iter", "eta_2"]].groupby("iter").agg(["mean", "var"])
        eta2_agg_by_iter.columns = ["mean", "var"]
        # Get var-between and -within
        eta2_VB = between_var(eta2_agg_by_iter["mean"], mean_eta2)
        eta2_VW = np.mean(eta2_agg_by_iter["var"])
        eta2_std_MI = np.sqrt(eta2_VB*(1 + 1/self.n_rep) + eta2_VW)
        # Save all
        eta2_results_dict = agg_results["eta_2"].to_dict()
        eta2_results_dict["std"] = eta2_std_MI
        eta2_results_dict["type"] = self.type
        eta2_results_dict = {k : eta2_results_dict[k] for k in ["mean","median","std","q025","q975","min","max","type"]}
        return pd.DataFrame(eta2_results_dict, index = [self.iter])

    def plugin_posterior_summarized(self):
        self.exposures = np.mean(self.results["expos"],axis=0)
        cur_data = {'X' : self.X, 'Z' : self.Z, 'Y' : self.Y,
                    'exposures' : self.exposures, 'adj_mat' : self.adj_mat}
        plugin_outcome_m = Outcome_MCMC(data=cur_data, n=self.n,
                                         type=self.type,
                                         rng_key=random.split(random.PRNGKey(0))[0],
                                         iter = self.iter,
                                         suff_stat=True)
        plugin_outcome_m.run_outcome_model()
        return plugin_outcome_m.get_summary_outcome_model()

    def get_results(self):
        if self.type in ["cut-2S","cut-3S"]:
            return self.cut_posterior_summarized()
        if self.type in ["plugin"]:
            return self.plugin_posterior_summarized()


if __name__ == "__main__":

    N = 300
    THETA = [-2, -0.5]
    GAMMA = [0.05, 0.25]
    ETA = [-1, -3, 0.5, -0.25]
    SIG_Y = 1
    PZ = 0.3
    BM_TYPES = ["cut-2S", "cut-3S", "plugin"]

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)

    # Get data
    df_true = DataGeneration(n=N, theta=THETA, eta=ETA, sig_y=SIG_Y, pz=PZ).get_data()
    # Generate noisy network measurement
    obs_network = create_noisy_network(df_true["adj_mat"], GAMMA, N)
    # save observed df and update A* and triu
    df_obs = df_true.copy()
    df_obs["adj_mat"] = obs_network["obs_mat"]
    df_obs["triu"] = obs_network["triu_obs"]

    # Run MCMC with true network (as given)
    oracle_outcome_mcmc = Outcome_MCMC(data=df_true, n=N, type = "oracle", rng_key=rng_key, iter = 1)
    oracle_outcome_mcmc.run_outcome_model()
    oracle_results = oracle_outcome_mcmc.get_summary_outcome_model()
    # Run MCMC with observed network (as given)
    obs_outcome_mcmc = Outcome_MCMC(data=df_obs, n=N, type = "observed", rng_key=rng_key, iter = 1)
    obs_outcome_mcmc.run_outcome_model()
    obs_results = obs_outcome_mcmc.get_summary_outcome_model()
    print(pd.concat([oracle_results, obs_results]))

    print("Running network model")
    # Run network module
    network_mcmc = Network_MCMC(data=df_obs, n=N, rng_key=rng_key)
    network_mcmc.run_network_model()
    print("Done with netowk model! getting predictive A* | . ")
    # get predictive distributions
    network_pred = network_mcmc.get_network_predictive(mean_posterior=False)
    network_pred_mean_post = network_mcmc.get_network_predictive(mean_posterior=True)

    print("Running estimation with cut-posterior and plugin")
    # Cut-posterior and plugin estimates
    # with 2S we need the `network_pred_mean_post`, and with the others the `network_pred` object
    # 2S
    cut_2S_mcmc = Bayes_Modular(data=df_obs, n=N, bm_type="cut-2S",
                                post_predictive=network_pred_mean_post, n_rep=20, iter=1)
    cut_2S_mcmc.run_inference()
    cut_2S_results = cut_2S_mcmc.get_results()
    # 3S
    cut_3S_mcmc = Bayes_Modular(data=df_obs, n=N, bm_type="cut-3S",
                                post_predictive=network_pred, n_rep=20, iter=1)
    cut_3S_mcmc.run_inference()
    cut_3S_results = cut_3S_mcmc.get_results()
    # plugin
    plugin_mcmc = Bayes_Modular(data=df_obs, n=N, bm_type="plugin",
                                post_predictive=network_pred, iter=1)
    plugin_mcmc.run_inference()
    plugin_results = plugin_mcmc.get_results()
    print(pd.concat([cut_2S_results, cut_3S_results,plugin_results]))
    print("ALL together:")
    results_all = pd.concat([oracle_results,obs_results,cut_2S_results,cut_3S_results,plugin_results])
    results_all['iter'] = 1
    print(results_all)
    print(results_all.to_string())



    # TODO: add class of `cut-posterior` (two and three stages) and `plug-in`
    # TODO: create simulation iterations procedure (in other py file perhaps)



