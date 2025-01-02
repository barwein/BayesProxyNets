import jax.numpy as jnp

a = jnp.array([0] * 100)
# a = 0
print(a, a.shape)
b = a.reshape(-1,1)
print(b, b.shape)

c = jnp.array([[0]*10, [1]*10]).T
print(c, c.shape)
d = c.T[None,:,:]
print(d, d.shape)

n, k = 5, 3
e = jnp.zeros((n, k))
print(e, e.shape)
e2 = jnp.zeros((10,))
print(e2, e2.shape)

import numpy as np

rng = np.random.default_rng(7)
ALPHAS = (1, 0)
N = 500

def stochastic_intervention(n_approx=1000, with_int=True):
    n_approx_s = n_approx if with_int else 1
    print("n_approx_s: ", n_approx_s)
    z_stoch1 = rng.binomial(n=1, p=ALPHAS[0], size=(n_approx_s, N))
    z_stoch2 = rng.binomial(n=1, p=ALPHAS[1], size=(n_approx_s, N))
    return jnp.array([z_stoch1, z_stoch2])

z_stoch = stochastic_intervention(45, True)
print(z_stoch, z_stoch.shape)
z_stoch2 = stochastic_intervention(45, False)
print(z_stoch2, z_stoch2.shape)


z_tst = jnp.array([[1]*N, [0]*N])
print(z_tst.shape)


dd = jnp.array([[1]*10, [0]*10])
print(dd, dd.shape)

dd2 = jnp.transpose(jnp.concatenate([dd, dd], axis=0))
print(dd2, dd2.shape)