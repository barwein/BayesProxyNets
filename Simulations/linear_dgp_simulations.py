import multiprocessing
import jax
import jax.numpy as jnp
import time
import numpy as np
from Simulations import results_to_pd_df, one_simuation_iter

# parameters guides:
# theta: p(A* | X, theta)
# gamma: p(A | A*, X, gamma)
# eta, sig_y: p(Y | Z, X, A*, eta, sig_y)
# alpha: pi_alpha(z) ---> stochastic intervention

# --- Define global variables ---
RANDOM_SEED = 62625235
rng = np.random.default_rng(RANDOM_SEED)

# parameters
THETA = [-2, 1.5]
GAMMA = [1.1, 0.2,  -1, 1]
ETA = jnp.array([-1, 3, -0.25, 0, 3])
SIG_Y = 1
PZ = 0.5
LIN_Y = False
ALPHAS = (0.7,0.3)
N = 500
N_SIM = 500
N_REP = 2000

### Note: this script is intended to run in a power-cluster and may take a while to finish ###
### Consider reducing N or N_SIM before running the simulation ###

if __name__ == "__main__":
    print("### Starting simulation ###")
    print("N_CORES: ", multiprocessing.cpu_count())
    print("N jax cpu devices: ", jax.local_device_count())
    start = time.time()
    idx_range = jnp.arange(N_SIM)
    # Run simulations
    for i in range(N_SIM):
        sim_results = one_simuation_iter(i, THETA, GAMMA, ETA, SIG_Y, PZ, N_REP, LIN_Y, ALPHAS)
        df_results = results_to_pd_df(sim_results, i)
        w_path = "results"
        dgp = "linear_dgp"
        with_header = i == 0
        df_results.to_csv(w_path + "/" + dgp + ".csv", index=False)
    print("Elapsed time: ", time.time()-start)













