#%%
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import jax


num_samples = 32
key = jax.random.PRNGKey(0)
a = jax.random.uniform(key, shape=(num_samples,), minval=-.8, maxval=.8)
x = jnp.linspace(-1, 1, 100)


# %%
output = jnp.where(x[None, :] < a[:, None], 0.0, 1.0)
# %%
fig, ax = plt.subplots()
sns.despine()
ax.plot(x, output.T, "C0", alpha=.3);
# %%
