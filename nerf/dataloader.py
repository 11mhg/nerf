import jax
import flax
import optax
import numpy as np
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
import random
from functools import cache

@cache
def generate_directions(height, width, focal, normalized = False):
    """
    Generates directions for a camera based on the height, width, focal length, and normalization flag.

    Parameters:
        height (int): The height of the camera.
        width (int): The width of the camera.
        focal (float): The focal length of the camera.
        normalized (bool, optional): Whether to normalize the directions. Defaults to False.

    Returns:
        np.ndarray: An array of directions in the shape (height, width, 3).
    """
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    i = i.astype(np.float32)
    j = j.astype(np.float32)

    if (normalized):
        i = (i / (width - 1) - 0.5) * 2
        j = (j / (height - 1) - 0.5) * 2
    
    transformed_i = (i - (width/2.)) / focal
    transformed_j = -(j - (height/2.)) / focal

    k = -np.ones_like(i)
    directions = np.stack((transformed_i, transformed_j, k), axis=-1)
    return directions

def generate_rays(height, width, focal, pose):
    """
    Generate rays based on the given parameters.

    Parameters:
        height (int): The height of the image.
        width (int): The width of the image.
        focal (float): The focal length of the camera.
        pose (ndarray): The pose matrix representing the camera position and orientation.

    Returns:
        dict: A dictionary containing the ray origins and directions.
            - 'origins': The origin points of the rays.
            - 'directions': The directions of the rays.
    """
    directions = generate_directions(height, width, focal)
    
    # Equivalent to: np.dot(directions, pose[:3, :3].T)
    # ray_directions = np.einsum("ijl,kl", directions, pose[:3, :3])
    ray_directions = directions @ pose[:3, :3].T
    ray_directions = ray_directions / jnp.linalg.norm(ray_directions, axis = -1, keepdims = True)

    ray_origins = np.broadcast_to(pose[:3, -1], ray_directions.shape)

    return {
        'origins': ray_origins.reshape(height*width, 3),
        'directions': ray_directions.reshape(height*width, 3)
    }
    
def stratified_sample(
    ray_origins,
    ray_directions,
    rng = None,
    near_bound = 2.,
    far_bound = 6.,
    num_samples = 256,
):
    """
    Generate stratified samples along rays using the origins and the directions of the rays. Must also define a near and far bound to defined camera bounds.

    Args:
        ray_origins (ndarray): The origins of the rays [bs, 3].
        ray_directions (ndarray): The directions of the rays [bs, 3].
        rng (RandomState, optional): The random state for generating noise. Defaults to None.
        near_bound (float, optional): The near bound for the sampling range. Defaults to 2.
        far_bound (float, optional): The far bound for the sampling range. Defaults to 6.
        num_samples (int, optional): The number of samples to generate. Defaults to 256.

    Returns:
        ndarray: The points along the rays [bs, num_samples, 3].
        ndarray: The corresponding t values [bs, num_samples].
    """
    t_vals = jnp.linspace(near_bound, far_bound, num_samples)
    t_vals = jnp.broadcast_to(t_vals, (ray_origins.shape[0], num_samples))

    if rng is not None:
        mid = (t_vals[:, :-1] + t_vals[:, 1:]) / 2.
        upper = jnp.concatenate([mid, t_vals[:, -1:]], axis = -1)
        lower = jnp.concatenate([t_vals[:, :1], mid], axis = -1)

        perturb_noise = jax.random.uniform(
            rng,
            t_vals.shape
        ) 
        
        t_vals = t_vals + (perturb_noise * (upper - lower))
        # noise = jax.random.uniform(
        #     rng,
        #     t_vals.shape
        # ) * ((far_bound - near_bound) / num_samples)
        # t_vals = t_vals + noise
    
    points = ray_origins[:,None,:] + (ray_directions[...,None, :] * t_vals[..., None])

    return points, t_vals


class Nerf_Data:
    def __init__(self, 
                 images, 
                 poses, 
                 focal, 
                 shuffle: bool = True, 
                 batch_size: int = 32,
                 num_samples = 256,
                 near_bound = 2.,
                 far_bound = 6.,
                 rng = None,
                 dtype = jnp.float32,
        ):
        self.images = images
        self.poses = poses
        self.focal = focal
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dtype = dtype
        self.rng = rng
        self.num_samples = num_samples
        self.near_bound = near_bound
        self.far_bound = far_bound
        self.ray_directions = []
        self.ray_origins = []
        self.prep_data()
    
    def prep_data(self):        
        height, width = self.images.shape[1:3]
        
        min_bound = jnp.array([100, 100, 100])
        max_bound = jnp.array([-100, -100, -100])

        for ind in range(self.poses.shape[0]):
            pose = self.poses[ind]
            rays = generate_rays(height, width, self.focal, pose)
            self.ray_directions.append(rays['directions'])
            self.ray_origins.append(rays['origins'])

            min_points = self.ray_origins[-1] + (self.near_bound * self.ray_directions[-1])
            max_points = self.ray_origins[-1] + (self.far_bound * self.ray_directions[-1])
            
            min_points = jnp.amin(min_points, axis = 0)
            max_points = jnp.amax(max_points, axis = 0)

            min_bound = jnp.minimum( min_points, min_bound)
            max_bound = jnp.maximum( max_points, max_bound)

        self.bounding_box = (
            min_bound - jnp.array([1, 1, 1]),
            max_bound + jnp.array([1, 1, 1])
        )

        self.inds = list(range(len(self.ray_origins)))
        return
    
    def __iter__(self):
        inds = self.inds
        if self.shuffle:
            random.shuffle(inds)
        
        for ind in inds:
            image = self.images[ind].reshape(-1, 3)
            ray_directions = self.ray_directions[ind]
            ray_origins = self.ray_origins[ind]
            
            if self.rng is not None:
                _, self.rng = jax.random.split(self.rng)

            points, t_vals = stratified_sample(
                ray_origins,
                ray_directions,
                rng = self.rng,
                near_bound = self.near_bound,
                far_bound = self.far_bound,
                num_samples = self.num_samples,
            )

            yield {
                'image': image,
                'position': points,
                'direction': ray_directions,
                't_vals': t_vals,
            }
            
        return
        #raise StopIteration

def get_dataloader( data_path: str, batch_size: int = 32, rng = jax.random.PRNGKey(0) ):
    """
    Generate a dataloader for the given data path.

    Args:
        data_path (str): The path to the data file.
        batch_size (int, optional): The batch size for the dataloader. Defaults to 32.
        rng (jax.random.PRNGKey, optional): The random seed for the dataloader. Defaults to jax.random.PRNGKey(0).

    Returns:
        Tuple[Nerf_Data, Nerf_Data, jnp.ndarray]: A tuple containing the train and test dataloaders and the bounding box bounds.
    """
    
    raw_data = np.load(data_path)
    images, poses = raw_data['images'], raw_data['poses']
    focal = float(raw_data['focal'])
    
    train_images, train_poses = images[:100], poses[:100]
    test_images, test_poses = images[100:], poses[100:]

    train_rng, test_rng = jax.random.split(rng)
        
    train_dl = Nerf_Data(train_images, train_poses, focal, rng = train_rng, batch_size = batch_size)
    test_dl = Nerf_Data(test_images, test_poses, focal, rng = test_rng, batch_size = batch_size)
    
    return train_dl, test_dl, train_dl.bounding_box

if __name__ == "__main__":
    
    train_dl, test_dl, bounding_box = get_dataloader('./tiny_nerf_data.npz')    
    
    max_pos, min_pos = 0, 0
    max_dir, min_dir = 0, 0
    for elem in train_dl:
        for key in elem:
            if key == 'position':
                max_pos = max(max_pos, elem[key].max())
                min_pos = min(min_pos, elem[key].min())
            elif key == 'direction':
                max_dir = max(max_dir, elem[key].max())
                min_dir = min(min_dir, elem[key].min())
    
    print(max_pos, min_pos)
    print(max_dir, min_dir)
