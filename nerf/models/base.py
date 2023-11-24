import jax
import flax 
import optax 
import numpy as np
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
from .utils import positional_encoding, calculate_alphas, calculate_accumulated_transformation


class Nerf(nn.Module):
    block_layers: tuple[int] = ( 5, 3 )
    block_width: int = 256
    dtype = jnp.float32
    precision = lax.Precision.DEFAULT
    
    # Following from nerf paper: https://arxiv.org/pdf/2003.08934.pdf
    position_encoding_dims: int = 10
    direction_encoding_dims: int = 4

    def get_learning_rate_schedule(self):
        return optax.exponential_decay(
            5e-4,
            50000,
            0.5,
            staircase = False,
            end_value = 5e-5
        )

    def get_optimizer(self, schedule = None):
        if schedule is None:
            schedule = self.get_learning_rate_schedule()
        return optax.chain(
            optax.adam(learning_rate = schedule),
        )
    
    @nn.compact
    def __call__(self, position: jnp.array, direction: jnp.array ):
        
        # pos_emb [bs, 3 + (3*2*self.position_encoding_dims)]
        # dir_emb [bs, 3 + (3*2*self.direction_encoding_dims)]
        pos_emb = positional_encoding(position, self.position_encoding_dims)
        dir_emb = positional_encoding(direction, self.direction_encoding_dims)

        y = pos_emb
        for ind, block_layer in enumerate(self.block_layers):
            for layer in range(block_layer):
                y = nn.Dense(
                    self.block_width, 
                    dtype = self.dtype,
                    precision = self.precision
                )(y)
                y = nn.relu(y)
            
            if ind == 0:
                y = jnp.concatenate([y, pos_emb], axis = -1)
        
        y = nn.Dense(
            self.block_width + 1,
            dtype = self.dtype,
            precision = self.precision
        )(y)

        y, sigma = y[...,:-1], y[...,-1:]
        sigma = nn.relu(sigma)

        y = jnp.concatenate([y, dir_emb], axis = -1)        
        y = nn.Dense(
            self.block_width//2,
            dtype = self.dtype,
            precision = self.precision
        )(y)
        y = nn.relu(y)
        
        rgb = nn.Dense(
            3,
            dtype = self.dtype,
            precision = self.precision
        )(y)
        rgb = nn.sigmoid(rgb)        
        return rgb, sigma
        

def initialize_model_variables(model, key: jax.random.PRNGKey):
    """
    Initializes the model variables using prng key.

    Args:
        model (Nerf): The model object.
        key (jax.random.PRNGKey): The random key.

    Returns:
        parameters (Dict): The initialized model parameters.
        key (jax.random.PRNGKey): The updated random key.
    """
    model_init_key, position_key, direction_key, key = jax.random.split(key, 4)
    dummy_position  = jax.random.uniform(position_key, (32, 256, 3), minval = -4., maxval = 4.)
    dummy_direction = jax.random.uniform(direction_key, (32, 256, 3), minval = -1., maxval = 1.)

    parameters = model.init( model_init_key, position = dummy_position, direction = dummy_direction  )
    return parameters, key

                    
if __name__ == "__main__":
    from nerf.train import render
    model = Nerf()

    key = jax.random.key(0)
    model_init_key, position_key, direction_key = jax.random.split(key, 3)

    dummy_position  = jax.random.uniform(position_key, (32, 256, 3), minval = -4., maxval = 4.)
    dummy_direction = jax.random.uniform(direction_key, (32, 3), minval = -1., maxval = 1.)
    dummy_direction = dummy_direction[:, None, :].repeat(dummy_position.shape[1], axis = 1)

    parameters = model.init( jax.random.key(0), position = dummy_position, direction = dummy_direction  )

    t_vals = jnp.linspace(2., 6., 256)
    t_vals = jnp.broadcast_to(t_vals, (32, 256))

    render(model, parameters, dummy_position, dummy_direction, t_vals)

    # rgb, sigma = model.apply(
    #     parameters,
    #     position = dummy_position,
    #     direction = dummy_direction
    # )

    # print(rgb.shape, sigma.shape)
