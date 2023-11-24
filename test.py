import jax
from jax import numpy as jnp
import numpy as np
from nerf.models.ngp import MultiResolutionHashEncoding, spherical_harmonic_encoding 

#a = np.random.rand(100, 3)
#a = (jnp.array(a) * 1000).astype(jnp.int32)
#
#bounding_box = (
#    jnp.array([-100, -100, -100], dtype=jnp.float32),
#    jnp.array([100, 100, 100], dtype=jnp.float32)
#)
#
#model = MultiResolutionHashEncoding(
#    bounding_box = bounding_box
#)
#
#rng = jax.random.PRNGKey(0)
#init_params_vars, x_vars, rng = jax.random.split(rng, 3)
#
#x_vars = jnp.array(
#    np.random.rand(100, 3)
#)
#
#params = model.init(
#    init_params_vars,
#    x = x_vars,
#) 

new_out = spherical_harmonic_encoding(jnp.array(np.random.rand(10000, 3)), 4)

print(new_out.shape)
