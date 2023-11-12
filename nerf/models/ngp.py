import jax
import flax
import optax
import numpy as np
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple
from .utils import positional_encoding, calculate_accumulated_transformation, calculate_alphas

def xor_reduce(array):
    """
    Performs an xor reduction on the last dimension of the given array.
    [..., C] => [..., 1]
    """
    results = jnp.zeros((*array.shape[:-1], 1), dtype=array.dtype)
    #def body(n, res):
    #    res = jnp.bitwise_xor(res[...,0], array[...,n:n+1])
    #    return res
    #results = lax.fori_loop(0, array.shape[-1], body, results)
    for i in range(array.shape[-1]):
        results = jnp.bitwise_xor( results[...,0:1], array[...,i:i+1] )
    return results


def hash_encode(inds, log2_hashmap_size):
    """
    Performs a cumulative xor reduction on the product of the indices and unique primes.
    This cumulative xor reduction then has a modulus operator done on it based on the log2_hashmap_size.
    """
    # First 3 values are from the paper
    primes = jnp.array([
        1,
        2654435761,
        805459861,
        720901,
        375391,
        432433,
        460157
    ])[None,...]

    primed_inds = (inds * primes[...,:inds.shape[-1]]).astype(jnp.int32)
    results = xor_reduce(primed_inds)
    return results % (2**log2_hashmap_size) 

BOX_OFFSETS = jnp.array(
    [[[i,j,k] for i in [0,1] for j in [0, 1] for k in [0,1]]]
)

def get_voxel_verts_from_xyz(
        xyz: jnp.ndarray, 
        bounding_box: Tuple[jnp.ndarray, jnp.ndarray], 
        resolution: float, 
        log2_hashmap_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    box_min, box_max = bounding_box
    mask = jnp.logical_and(
        jnp.all(
            xyz >= box_min, axis=-1
        ),
        jnp.all(
            xyz <= box_max, axis=-1
        )
    )
    grid_size = (box_max - box_min) / resolution

    bottom_left_idx = jnp.floor((xyz-box_min)/grid_size).astype(jnp.int32)
    voxel_min_vert = bottom_left_idx*grid_size + box_min
    voxel_max_vert = voxel_min_vert + grid_size
    voxel_indices = bottom_left_idx[..., None, :] + BOX_OFFSETS

    hashed_voxel_indices = hash_encode(voxel_indices, log2_hashmap_size)[...,0]

    return voxel_min_vert, voxel_max_vert, hashed_voxel_indices, mask


class MultiResolutionHashEncoding(nn.Module):

    bounding_box: Tuple[jnp.ndarray, jnp.ndarray]
    encoding_levels: int = 16
    hash_feature_dim: int = 2
    log2_hashmap_size: int = 19
    base_resolution: int = 16
    per_level_scale: int = 2


    def setup(self):

        self.finest_resolution = int(self.base_resolution * (self.per_level_scale) ** (self.encoding_levels-1))
        self.out_dim = self.encoding_levels * self.hash_feature_dim
        self.b = jnp.exp( (jnp.log(self.finest_resolution) - jnp.log(self.base_resolution)) / (self.encoding_levels-1) )

        embeds = []
        for i in range(self.encoding_levels):
            embeds.append(
                nn.Embed(
                    2**self.log2_hashmap_size, 
                    self.hash_feature_dim,
                )
            )
        self.embeds = embeds

        return

    def trilinear_interp(self, x, vx_min, vx_max, vx_emb):

        weights = (x - vx_min) / (vx_max - vx_min) # renormalize using trilinear interp func 
        inv_weights = 1. - weights

        c00 = (vx_emb[:,0] * inv_weights[:,0:1]) + (vx_emb[:, 4] * weights[:,0:1])
        c01 = (vx_emb[:,1] * inv_weights[:,0:1]) + (vx_emb[:, 5] * weights[:,0:1])
        c10 = (vx_emb[:,2] * inv_weights[:,0:1]) + (vx_emb[:, 6] * weights[:,0:1])
        c11 = (vx_emb[:,3] * inv_weights[:,0:1]) + (vx_emb[:, 7] * weights[:,0:1])

        c0 = c00*inv_weights[:, 1:2] + c10*weights[:,1:2]
        c1 = c01*inv_weights[:, 1:2] + c11*weights[:,1:2]

        c = c0*inv_weights[:,2:3] + c1*weights[:,2:3]
        return c
    
    def __call__(self, x):
        embeds = []
        for i in range(self.encoding_levels):
            resolution = jnp.floor( self.base_resolution * (self.b ** i))
            voxel_min_vert, voxel_max_vert, hashed_vox_inds, mask = get_voxel_verts_from_xyz(
                x, self.bounding_box, resolution, self.log2_hashmap_size
            )
            voxel_embeds = self.embeds[i]( hashed_vox_inds ) # Should return the embeddings for the hashed voxel inds

            enc_x = self.trilinear_interp( x, voxel_min_vert, voxel_max_vert, voxel_embeds )
            embeds.append(enc_x)
        embeds = jnp.concatenate(
            embeds,
            axis = -1
        )
        return embeds


def spherical_harmonic_encoding(xyz, n: int = 4):
        result = [jnp.array([ 0.5 * jnp.sqrt(1. / jnp.pi)]).reshape(1, 1).repeat(xyz.shape[0], 0)]

        x = xyz[...,0:1]
        y = xyz[...,1:2]
        z = xyz[...,2:3]

        if n >= 1:
            result.append( 
                -0.5 * np.sqrt( 3. / np.pi) * y,  # -1/2*sqrt(3/pi) * y
            )
            result.append(
                 0.5 * np.sqrt( 3. / np.pi) * z,  #  1/2*sqrt(3/pi) * z 
            )
            result.append(
                -0.5 * np.sqrt( 3. / np.pi) * x,  # -1/2*sqrt(3/pi) * x
            )
        
        x2 = x**2.
        y2 = y**2. 
        z2 = z**2.
        xy = x*y
        xz = x*z
        yz = y*z
        
        if n >= 2:

            subtractor = 1/4 * np.sqrt(5/np.pi)
            v1 = 1/4 * np.sqrt(15/np.pi)
            v2 = 1/2 * np.sqrt(15/np.pi)
            v3 = 1/4 * np.sqrt(5/np.pi) * 3

            result = [
                *result,
                v2 * xy,
                -v2 * yz,
                (v3 * z2) - subtractor,
                -v2 * xz,
                (v1 * x2) - (v1 * y2),
            ]
        
        if n >= 3:
            v1 = (1/4) * np.sqrt(105/np.pi) # 1.445305721
            v2 = (1/2) * np.sqrt(105/np.pi) # 2.8906...
            v3 =  1/4 * np.sqrt(35 / (2 * np.pi)) # 0.5900435899
            v4 = (1/2) * np.sqrt( 7 / (6 * np.pi)) # 0.304697199

            result = [
                *result,
                -v3 * y * (3. * x2 - y2),
                v2 * xy * z,
                v4 * y * (1.5 - 7.5 * z2),
                1.24392110863372 * z * (1.5 * z2 - 0.5) - 0.497568443453487 * z,
                v4 * x * (1.5 - 7.5 * z2),
                v1 * z * (x2 - y2),
                -v3 * x * (x2 - 3. * y2),
            ]

        if n >= 4:
            x4 = x**4 
            y4 = y**4

            result = [
                *result,
                2.5033429417967 * xy * (x2 - y2),
                -1.77013076977993 * yz * (3.0 * x2 - y2),
                0.126156626101008 * xy * (52.5 * z2 - 7.5),
                0.267618617422916 * y * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z),
                1.48099765681286
                * z
                * (1.66666666666667 * z * (1.5 * z2 - 0.5) - 0.666666666666667 * z)
                - 0.952069922236839 * z2
                + 0.317356640745613,
                0.267618617422916 * x * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z),
                0.063078313050504 * (x2 - y2) * (52.5 * z2 - 7.5),
                -1.77013076977993 * xz * (x2 - 3.0 * y2),
                -3.75501441269506 * x2 * y2
                + 0.625835735449176 * x4
                + 0.625835735449176 * y4,
            ]
        
        return np.concatenate(result, axis = -1) 



class NerfNGP(nn.Module):

    # Encoding stuff
    bounding_box: Tuple[jnp.ndarray, jnp.ndarray]
    encoding_levels: int = 16
    hash_feature_dim: int = 2
    log2_hashmap_size: int = 19
    base_resolution: int = 16
    per_level_scale: int = 2

    color_mlp_hidden_layers: int = 2
    density_mlp_hidden_layers: int = 1
    density_out_dim: int = 16
    color_out_dim: int = 3
    width: int = 64
    harmonic_encoding_degree:int = 4
    dtype = jnp.float32
    precision = lax.Precision.DEFAULT

    def get_learning_rate_schedule(self):
        return optax.exponential_decay(
            1e-2,
            50000,
            0.5,
            staircase = False,
            end_value = 1e-4
        )

    def get_optimizer(self, schedule = None):
        if schedule is None:
            schedule = self.get_learning_rate_schedule
        return optax.chain(
            optax.adam(
                learning_rate = schedule,
                b1 = 0.9,
                b2 = 0.99,
                eps = 1e-15
            )
        )
    
    @nn.compact
    def __call__(self, position: jnp.array, direction: jnp.array ):
        
        original_shape = position.shape
        if (len(position.shape) > 2):
            position = position.reshape( -1, position.shape[-1] )
            direction = direction.reshape( -1, direction.shape[-1] )

        encoded_position = MultiResolutionHashEncoding(
            bounding_box      = self.bounding_box,
            encoding_levels   = self.encoding_levels,
            hash_feature_dim  = self.hash_feature_dim,
            log2_hashmap_size = self.log2_hashmap_size,
            base_resolution   = self.base_resolution,
            per_level_scale   = self.per_level_scale
        )(position)
        encoded_direction = spherical_harmonic_encoding(direction, self.harmonic_encoding_degree)

        # density_mlp_layers
        y = encoded_position 
        for _ in range(self.density_mlp_hidden_layers):
            y = nn.Dense(self.width, dtype=self.dtype, precision = self.precision)(y)
            y = nn.relu(y)
        density_output = nn.Dense(self.density_out_dim, dtype = self.dtype, precision = self.precision)(y)

        sigma = nn.relu(density_output[...,-1:])

        # color_mlp_layers
        y = jnp.concatenate(
            [
                density_output,
                encoded_direction,
            ],
            axis = -1 
        )
        for _ in range(self.color_mlp_hidden_layers):
            y = nn.Dense(self.width, dtype=self.dtype, precision=self.precision)(y)
            y = nn.relu(y)
        rgb_output = nn.Dense(self.color_out_dim, dtype=self.dtype, precision=self.precision)(y)

        rgb = nn.sigmoid(rgb_output)

        # Reshape back to normal
        rgb = rgb.reshape(*original_shape)
        sigma = sigma.reshape(*original_shape[:-1], 1)

        return rgb, sigma
        

def initialize_model_variables(model: NerfNGP, key: jax.random.PRNGKey):
    """
    Initializes the model variables using prng key.

    Args:
        model (NerfNGP): The model object.
        key (jax.random.PRNGKey): The random key.

    Returns:
        parameters (Dict): The initialized model parameters.
        key (jax.random.PRNGKey): The updated random key.
    """
    model_init_key, position_key, direction_key, key = jax.random.split(key, 4)
    dummy_position  = jax.random.uniform(position_key, (32 * 256, 3), minval = -4., maxval = 4.)
    dummy_direction = jax.random.uniform(direction_key, (32 * 256, 3), minval = -1., maxval = 1.)

    parameters = model.init( model_init_key, position = dummy_position, direction = dummy_direction  )
    return parameters, key

                    
if __name__ == "__main__":
    from nerf.train import render
    dummy_bounding_box = (
        jnp.array([-100,-100,-100], dtype=jnp.float32),
        jnp.array([100,100,100], dtype=jnp.float32)
    )
    model = NerfNGP(bounding_box = dummy_bounding_box)

    key = jax.random.key(0)
    model_init_key, position_key, direction_key = jax.random.split(key, 3)

    dummy_position  = jax.random.uniform(position_key, (32, 256, 3), minval = -4., maxval = 4.)
    dummy_direction = jax.random.uniform(direction_key, (32, 256, 3), minval = -1., maxval = 1.)

    parameters = model.init( jax.random.key(0), position = dummy_position, direction = dummy_direction  )

    rgb, sigma = model.apply( parameters, position = dummy_position, direction = dummy_direction)

    print(rgb.shape, sigma.shape)

    #t_vals = jnp.linspace(2., 6., 256)
    #t_vals = jnp.broadcast_to(t_vals, (32, 256))

    #render(model, parameters, dummy_position, dummy_direction, t_vals)

    # rgb, sigma = model.apply(
    #     parameters,
    #     position = dummy_position,
    #     direction = dummy_direction
    # )

    # print(rgb.shape, sigma.shape)
