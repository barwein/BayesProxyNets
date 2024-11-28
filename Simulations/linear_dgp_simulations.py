import multiprocessing
import jax
import jax.numpy as jnp
import time
import numpy as np
import src.Aux_functions as aux
from Simulations.Simulations import results_to_pd_df, one_simuation_iter


# parameters guides:
# theta: p(A* | X, theta)
# gamma: p(A | A*, X, gamma)
# eta, sig_y: p(Y | Z, X, A*, eta, sig_y)
# alpha: pi_alpha(z) ---> stochastic intervention

# --- Define global variables ---
RANDOM_SEED = 515
rng = np.random.default_rng(RANDOM_SEED)

# parameters
# THETA = [-1.5, 1.5]
THETA = [-2, 1.5]
GAMMA = [1.1, 0.2, -1, 1]
GAMMA_REP = [1, 0.7, -1, 0.6]
# ETA = jnp.array([-1, 3, -0.25, 0, 3])
ETA = jnp.array([-1, 3, -0.25, 3])
SIG_Y = 1.5
# SIG_Y = 1
RHO = 0.9
PZ = 0.5
LIN_Y = True
ALPHAS = (0.7, 0.3)
N = 500
# N = 100
# N_SIM = 500
N_SIM = 2
# N_REP = 2000
N_REP = 30

### Note: this script is intended to run in a power-cluster and may take a while to finish ###
### Consider reducing N or N_SIM before running the simulation ###

if __name__ == "__main__":
    print("### Starting simulation ###")
    print("N_CORES: ", multiprocessing.cpu_count())
    print("N jax cpu devices: ", jax.local_device_count())

    # Get fixed data across all simulations
    fixed_df = aux.GenerateFixedData(
        rng=rng, theta=THETA, eta=ETA, sig_y=SIG_Y, lin=LIN_Y, alphas=ALPHAS, n=N
    ).get_data()

    start = time.time()
    idx_range = jnp.arange(N_SIM)

    # Run simulations
    for i in range(N_SIM):
        sim_results = one_simuation_iter(
            idx=i,
            fixed_df=fixed_df,
            gamma=GAMMA,
            gamma_rep=GAMMA_REP,
            eta=ETA,
            sig_y=SIG_Y,
            rho=RHO,
            pz=PZ,
            n_rep=N_REP,
            lin_y=LIN_Y,
        )
        df_results = results_to_pd_df(sim_results, 1)
        w_path = "results"
        dgp = "linear_dgp_test"
        with_header = i == 0
        df_results.to_csv(
            w_path + "/" + dgp + ".csv", index=False, mode="a", header=with_header
        )
    print("Elapsed time: ", time.time() - start)
