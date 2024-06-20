import os
import jax
import jax.numpy as jnp
import multiprocessing
import pandas as pd

n_cores = multiprocessing.cpu_count()
print("N_CORES: ", n_cores)
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(n_cores)
print(jax.devices('cpu'))

def test_func(idx, x, y):
    a1 = jnp.array([idx, x, y])
    a2 = jnp.array([-1,y,x])
    return jnp.vstack([a1,a2])

test_vmap = jax.vmap(test_func, in_axes=(0,None,None))

res = test_vmap(jax.numpy.arange(4), 3, 4)
concatenated_res = jnp.vstack(res)

print(concatenated_res)

column_names = ["index", "value1", "value2"]
res_c = pd.DataFrame(concatenated_res, columns=column_names)
print(res_c)
