from os.path import isdir
import jax
import math
import orbax
import optax
import os, sys
import argparse
import functools
import rerun as rr
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint
from flax.training.train_state import TrainState

from nerf.dataloader import generate_rays, stratified_sample
from nerf.model import Nerf, initialize_model_variables, calculate_alphas
from nerf.train import render


#def generate_directions(height, width, focal, normalized = False, step_size = 1.):
#    i,j = np.meshgrid(np.arange(0, width, step_size), np.arange(0, height, step_size), indexing='xy')
#    i = i.astype(np.float32)
#    j = j.astype(np.float32)
#
#    if (normalized):
#        i = (i / (width - 1) - 0.5) * 2
#        j = (j / (height - 1) - 0.5) * 2
#
#    transformed_i = (i - (width/2.)) / focal
#    transformed_j = -(j - (height/2.)) / focal
#
#    k = -np.ones_like(i)
#    directions = np.stack((transformed_i, transformed_j, k), axis=-1)
#    return directions
#
#def generate_rays(height, width, focal, pose, step_size = 1.):
#    directions = generate_directions(height, width, focal, step_size = step_size)
#    
#    # Equivalent to: np.dot(directions, pose[:3, :3].T)
#    # ray_directions = np.einsum("ijl,kl", directions, pose[:3, :3])
#    ray_directions = directions @ pose[:3, :3].T
#    ray_directions = ray_directions / jnp.linalg.norm(ray_directions, axis = -1, keepdims = True)
#
#    ray_origins = np.broadcast_to(pose[:3, -1], ray_directions.shape)
#
#    return {
#        'origins': ray_origins.reshape(-1, 3),
#        'directions': ray_directions.reshape(-1, 3)
#    }


def initialize_model_from_path(model_path):
    model = Nerf()
    params, rng = initialize_model_variables(model, jax.random.PRNGKey(0))
    
    optimizer = optax.chain(
        optax.adam(learning_rate = 1e-5),
    )
    state = TrainState.create(
        apply_fn = model.apply,
        params = params,
        tx = optimizer
    )
    target = {'state': state, 'best_loss': np.Inf}
    
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=3, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        model_path, orbax_checkpointer, options)

    
    def load_fn(step, target):
        ckpt = checkpoint_manager.restore(step, items=target )
        state = ckpt['state']
        best_loss = ckpt['best_loss']
        return state, best_loss

    state, best_loss = load_fn(checkpoint_manager.latest_step(), target)
    print(f"Restored model from step {checkpoint_manager.latest_step()}")
    
    return  model, state.params

def get_poses(data_path):
    raw_data = np.load(data_path)
    poses = raw_data['poses']
    focal = float(raw_data['focal'])
    return poses, focal

def visualize_rays(ray_origins, ray_directions):
    rr.log(
        "init/pinhole/position", 
        rr.Points3D(ray_origins[:1], radii = 1e-2,  colors = (0, 255, 0)),
    )
    rr.log(
        "init/pinhole/direction",
        rr.Arrows3D(
            origins = np.reshape(ray_origins[::10], [-1, 3]), 
            vectors = np.reshape(ray_directions[::10], [-1, 3]),
            radii = 1e-3,
            colors = (0, 0, 255)
        )
    )

def sample_from_origin_and_direction(origin, directions, num_samples=256):
    origin = np.reshape(origin, [-1, 3])
    origin = origin.repeat(directions.shape[0], axis = 0)

    points, t_vals = stratified_sample(
        origin,
        directions,
        rng = None,
        near_bound = 2.,
        far_bound = 6.,
        num_samples = num_samples,
    )

    return points, t_vals

def get_point_cloud( model, params, position, direction, t_vals):
    rgb, sigma = model.apply(
        params,
        position = position,
        direction = direction,
    )

    rgb = jnp.broadcast_to(rgb, position.shape) # [bs, num_samples, 3]

    deltas = jnp.reshape( t_vals, (-1, t_vals.shape[-1]) )
    deltas_inf = jnp.array(
        [[1e10]],
    ).repeat(deltas.shape[0], axis = 0)
    deltas = jnp.concatenate(
        (deltas[...,1:] - deltas[...,:-1], deltas_inf),
        axis = -1
    )
    alphas = calculate_alphas( sigma[...,0], deltas )
   
    rgba = jnp.concatenate(
        [
            rgb, alphas[...,None]
        ],
        axis = -1
    )
    return rgba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type = dir_path, help = "The path to the model directory" 
    )
    parser.add_argument(
        "--data", type = file_path, help = "The path to the data file containing poses and rotations."
    )
    parser.add_argument(
        "--image_shape", type = str, help = "The shape of the image to render. Defaults to [100,100]",
        default = "100,100"
    )
    parser.add_argument(
        "--num_samples", type = int, help = "The number of samples to render. Defaults to 256",
        default = 256
    )

    args = parser.parse_args()
    
    model_weights_path = args.model
    data_path = args.data
    image_shape = args.image_shape.split(",")
    height = int(image_shape[0])
    width = int(image_shape[1])
   
    rr.init("rerun_nerf", spawn = True)

    model, params = initialize_model_from_path(model_weights_path)
    
    poses, focal = get_poses(data_path)
    starting_pose = poses[0]
    rays = generate_rays(height, width, focal, starting_pose)
    
    height, width = int(np.sqrt(rays['origins'].shape[0])), int(np.sqrt(rays['origins'].shape[0]))
    
    position = rays['origins'][0]
    direction = rays['directions']

    visualize_rays(rays['origins'], direction)
    points, t_vals = sample_from_origin_and_direction(position, direction, num_samples = args.num_samples)
    
    rr.log(
        "init/camera_bounds/points",
        rr.Points3D(
            points[::25, ::4].reshape((-1,3)),
        )
    )
    
    render_fn = functools.partial( render,
        params = params,
        model = model
    )
    cloud_point_fn = functools.partial(
        get_point_cloud,
        model = model,
        params = params
    )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _generate_image(render_fn, points, direction, t_vals):
        direction = direction[:,None,:].repeat( points.shape[1], axis = 1)
        colours, _, _, _ = render_fn(
            position = points,
            direction = direction,
            t_vals = t_vals
        )
        return colours

    def generate_image(render_fn, points, direction, t_vals, batch_size = 1024):
        imgs = []
        num_batches = math.ceil( points.shape[0] / batch_size )
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            img = _generate_image(render_fn, points[start_idx:end_idx], direction[start_idx:end_idx], t_vals[start_idx:end_idx])
            imgs.append(img)
        imgs = jnp.concatenate(imgs, axis = 0)
        return jnp.reshape( imgs, (height, width, 3) )

    points = jnp.array(points)
    direction = jnp.array(direction)
    t_vals = jnp.array(t_vals)
    colours = generate_image(render_fn, points, direction, t_vals)

    rr.log(
        "camera/pinhole",
        rr.Pinhole( focal_length = focal, height = height, width = width ),
    )
    rr.log(
        "camera/pinhole",
        rr.Image(colours),
    )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _generate_point_cloud(cloud_point_fn, points, direction, t_vals):
        direction = direction[:,None,:].repeat( points.shape[1], axis = 1)
        point_cloud = cloud_point_fn(
            position = points,
            direction = direction,
            t_vals = t_vals
        )
        return point_cloud
    
    def generate_point_cloud(cloud_point_fn, points, direction, t_vals, batch_size = 1024):
        point_clouds = []
        num_batches = math.ceil( points.shape[0] / batch_size )
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            point_cloud = _generate_point_cloud(cloud_point_fn, points[start_idx:end_idx], direction[start_idx:end_idx], t_vals[start_idx:end_idx])
            point_clouds.append(point_cloud)
        point_clouds = jnp.concatenate(point_clouds, axis = 0)
        return point_clouds
    
    if True:
        rgba = generate_point_cloud(cloud_point_fn, points, direction, t_vals)
        
        rgba = np.array(rgba).reshape(-1, 4)
        np_points = np.array(points).reshape(-1, 3)
        filter_mask = rgba[:, 3] > 0.1
        rgba = rgba[filter_mask]
        np_points = np_points[filter_mask]
        
        rr.log("render/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless = True)
        rr.log(
            "render/world/point_cloud",
            rr.Points3D(
                np_points[::8],
                colors = rgba[::8],
            )
        )

    return

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("Dir:{0} is not a valid path".format(path))

def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError("File:{0} is not a valid path".format(path))

if __name__ == "__main__":

    main()

