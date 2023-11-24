import jax
import flax 
import orbax
import numpy as np
from jax import lax
import orbax.checkpoint
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple

#@jax.jit
def positional_encoding(position, dims):
    pos_enc = [position]
    for L in range(dims):
        pos_enc.append(jnp.sin(2.**L * jnp.pi * position))
        pos_enc.append(jnp.cos(2.**L * jnp.pi * position))
        # pos_enc.append(jnp.sin(2.*jnp.pi*position/L))
        # pos_enc.append(jnp.cos(2.*jnp.pi*position/L))
    pos_enc = jnp.concatenate(pos_enc, axis = -1)
    return pos_enc


def calculate_accumulated_transformation(alphas):
    """
    Calculate the accumulated transformation using the given alphas.
    
    This function has a bit of a hack to deal with the fact that it may not
    accumulate to 1. Instead, it forces it by setting the final value to 1.
    """
    # Calculate the cumulative product of 1 minus alphas
    acc_trans = jnp.cumprod(1. - alphas, axis=-1)
    
    # Concatenate a column of ones to the left of acc_trans
    return jnp.concatenate((jnp.ones((acc_trans.shape[0], 1)), acc_trans[..., :-1]), axis=-1)


def calculate_alphas(sigmas, deltas):
    """Given sigmas and sampling point deltas, calculate the alphas

    Args:
        sigmas (jnp.array): The sigmas describing the density of the points sampled.
        deltas (jnp.array): The sampling point deltas.

    Returns:
        (jnp.array): The alphas describing the density of the points sampled 
                     after applying the deltas.
    """
    return (1. - jnp.exp(-sigmas * deltas))


def get_checkpoint_manager(save_path, max_to_keep: int = 2, create: bool = True):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep = max_to_keep, create = create)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        save_path, orbax_checkpointer, options
    )
    return checkpoint_manager
