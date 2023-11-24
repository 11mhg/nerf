import os
import matplotlib.pyplot as plt
import cv2
import math
import functools
import jax
import flax
import optax
import random
import numpy as np
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
import scipy.spatial.distance
import random
from functools import cache
from nerf.dataloader import Nerf_Data, generate_rays, get_dataloader
from nerf.train import render
from nerf.models.base import initialize_model_variables
from nerf.models.base import Nerf
from nerf.models.ngp import NerfNGP
from nerf.models.utils import get_checkpoint_manager
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import orbax.checkpoint
from prefetch_generator import BackgroundGenerator 


def sample_random_starting_point(scale = np.array([4., 4., 4.])):
    x = np.random.normal(0., 1., size=(1000))
    y = np.random.normal(0., 1., size=(1000))
    z = np.random.normal(0., 1., size=(1000))
    vec = np.stack([x, y, z], axis = 1)
    vec = vec / np.linalg.norm(vec, axis = 1)[:, None]
    vec[...,2] = np.abs(vec[...,2])
    vec = vec * scale.reshape(-1, 3)
    
    dists = scipy.spatial.distance.cdist(vec, vec)
    max_inds = np.unravel_index(dists.argmax(), dists.shape)

    starting_point = vec[max_inds[0]]
    ending_point = vec[max_inds[1]]

    mid_point_ind = random.choice( [ i for i in range(vec.shape[0]) if i not in max_inds ] )
    mid_point = vec[mid_point_ind]

    return starting_point, mid_point, ending_point

def move_to_point_in_seconds(starting_point, ending_point, seconds = 5, fps = 24):
    total_number_of_steps = 5 * 24
    step_size = (ending_point - starting_point)/total_number_of_steps
    all_points = [starting_point]
    for _ in range(total_number_of_steps):
        all_points.append(all_points[-1] + step_size)
    return np.stack(all_points, axis = 0)


def direction_vector_to_euler(directions):
    directions /= np.linalg.norm(directions, axis = 1)[:, None]
    upVector = np.array([[0, 0, 1]])
    rightVector = np.array([[0, 1, 0]])
    forwardVector = np.array([[1, 0, 0]])
        
    yaw = directions
    yaw[...,2] = 0. 
    yaw = np.arctan2(yaw[:,1], yaw[:,0])
    pitch = np.arcsin(-directions[...,2])
    roll = np.zeros_like(pitch)
    euler = np.stack([roll, pitch, yaw], axis = -1)
    return euler

def euler_to_R(euler):
    Rz = np.array([
        [ np.cos(euler[:,2])       ,         -np.sin(euler[:,2]), np.zeros_like(euler[:,2]) ],
        [ np.sin(euler[:,2])       ,          np.cos(euler[:,2]), np.zeros_like(euler[:,2]) ],
        [ np.zeros_like(euler[:,2]),   np.zeros_like(euler[:,2]), np.zeros_like(euler[:,2]) ],
    ]).transpose(2, 0, 1)

    Rx = np.array([
        [ np.ones_like(euler[:,1]), np.zeros_like(euler[:,1]), np.zeros_like(euler[:,1]) ],
        [ np.zeros_like(euler[:,1]), np.cos(euler[:,1]), -np.sin(euler[:,1]) ],
        [ np.zeros_like(euler[:,1]), np.sin(euler[:,1]), np.cos(euler[:,1]) ],
    ]).transpose(2, 0, 1)

    Ry = np.array([
        [ np.cos(euler[:,0]), np.zeros_like(euler[:,0]), np.sin(euler[:,0]) ],
        [ np.zeros_like(euler[:,0]), np.ones_like(euler[:,0]), np.zeros_like(euler[:,0]) ],
        [ -np.sin(euler[:,0]), np.zeros_like(euler[:,0]), np.cos(euler[:,0]) ]
    ]).transpose(2, 0, 1)

    I = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    I = I.reshape(1, 3, 3).repeat(Rz.shape[0],0)
    
    R = np.cross(Rz, I)
    R = np.cross(Rx, R)
    R = np.cross(Ry, R)

    return R

#def generate_path_and_directions(train_dl):
#    ray_origins = train_dl.ray_origins
#    ray_directions = train_dl.ray_directions
#
#    center_position = np.mean(np.stack(ray_origins, axis = 0)[:, 0, :], axis = 0)
#    center_position[...,2] = 0.
#
#    starting_point, mid_point, end_point = sample_random_starting_point()
#
#    origins = move_to_point_in_seconds(starting_point, mid_point)
#    origins = np.concatenate([origins, move_to_point_in_seconds(mid_point, end_point)], axis = 0)
#   
#    mean_directions = center_position[None, :] - origins
#    mean_directions_norm = mean_directions / (np.linalg.norm(mean_directions) + 1e-16)
#    eulers = direction_vector_to_euler(mean_directions_norm)
#    pose = euler_to_R(eulers)
#    
#    pose = np.concatenate([ pose, origins[:, :, None] ], axis = -1)
#    
#    for ind in range(pose.shape[0]):
#        info = generate_rays(100, 100, train_dl.focal, pose[ind])
#        yield info, ind, pose.shape[0]

    #return origins, directions, train_dl.focal

def pose_spherical(theta, phi, roll, t):
    pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ])
    phi = phi / 180. * np.pi
    pose = np.array([
        [ 1, 0, 0, 0 ],
        [ 0, np.cos(phi), - np.sin(phi), 0 ],
        [ 0, np.sin(phi),   np.cos(phi), 0 ],
        [ 0, 0, 0, 1 ]
    ]) @ pose

    theta = theta / 180. * np.pi
    pose = np.array([
        [np.cos(theta), 0, -np.sin(theta), 0],
        [            0, 1,              0, 0],
        [np.sin(theta), 0,  np.cos(theta), 0],
        [            0, 0,              0, 1]
    ]) @ pose

    roll = roll / 180. * np.pi
    pose = np.array([
        [-1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1]
    ]) @ pose
    
    return pose



def generate_path_and_directions(train_dl):
    thetas = np.linspace(0., 360., 180, endpoint = False)# List 180 thetas from 0 to 360
    for ind, theta in enumerate(thetas):
        camera_to_world = pose_spherical(theta, -30., 0., 4.)

        info = generate_rays(100, 100, train_dl.focal, camera_to_world)
        yield info, ind, thetas.shape[0] 

def generate_video(
        model: Nerf, 
        params: dict, 
        train_dl: Nerf_Data, 
        batch_size: int=512, 
        save_path:str = './model-ckpt', 
        model_name:str = 'nerf'
    ):
    
    checkpoint_manager = get_checkpoint_manager(save_path, max_to_keep = 2, create = False)
    schedule = model.get_learning_rate_schedule()

    optimizer = model.get_optimizer(schedule)
    state = TrainState.create(
        apply_fn = functools.partial( render, model = model),
        params = params,
        tx = optimizer
    )
    
    save_args = orbax_utils.save_args_from_target({'state': state, 'best_loss': np.Inf})

    if checkpoint_manager.latest_step() is None:
        raise ValueError("No found saved weights")

    def load_fn(step, state, best_loss = np.Inf):
        target = { 'state': state, 'best_loss': best_loss }
        ckpt = checkpoint_manager.restore(step, items=target)
        state = ckpt['state']
        best_loss = ckpt['best_loss']
        return state, best_loss

    state, best_loss = load_fn(checkpoint_manager.latest_step(), state)
    print(f"Found model with loss {best_loss}")

    info_gen = generate_path_and_directions(train_dl) 

    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
    out = cv2.VideoWriter(os.path.join( save_path, 'video.avi' ), 
        cv2.VideoWriter_fourcc(*"MJPG"), 24., (100, 100))

    for info, ind, total in BackgroundGenerator(info_gen):
        print(f"Generating frame {ind+1}/{total}")
        generated_image = []
        ray_origins = info['origins']
        ray_directions = info['directions']

        t_vals = jnp.linspace( train_dl.near_bound, train_dl.far_bound, train_dl.num_samples )
        points = ray_origins[:, None, :] + (ray_directions[...,None, :] * t_vals[...,None])
        directions = jnp.array(ray_directions)

        num_batches = math.ceil( points.shape[0] / batch_size )
        for batch_ind in range(num_batches):
            batch_points = points[(batch_ind * batch_size):((batch_ind+1) * batch_size)]
            batch_directions = directions[(batch_ind * batch_size):((batch_ind+1) * batch_size)]
            batch_directions = batch_directions[:,None,:].repeat(
                batch_points.shape[1],
                axis = 1
            )
            batch_t_vals = t_vals

            colors, _, _, _ = state.apply_fn(
                params = state.params,
                position = batch_points,
                direction = batch_directions,
                t_vals = batch_t_vals
            )
            generated_image.append(colors)
        
        generated_image = jnp.concatenate(generated_image, axis = 0)
        generated_image = jnp.asarray(generated_image)
        generated_image = np.array(generated_image)
        generated_image = np.reshape(
            generated_image,
            (100, 100, 3)
        )
        generated_image = np.clip(generated_image * 255., 0., 255.).astype(np.uint8)[...,::-1]
        cv2.imwrite(
            os.path.join( save_path, 'images', f"{ind+1}.jpg" ), generated_image
        )
        out.write(generated_image)
    out.release()
    print("Done Writing video")
    

    return 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog = "GenVid",
        description = "Generate video from random path"
    )
    parser.add_argument('--dataset', '-d', type=str, help="The dataset used for training the nerf model",
        default="./tiny_nerf_data.npz"
    )
    parser.add_argument('--model', '-m', type=str, help="The model to use: nerf or ngp. Defaults to nerf.",
        default = "nerf"
    )
    parser.add_argument('--save-path', '-s', type=str, help = "The save path for the model checkpoints",
        default = "./model-ckpt"
    )
    parser.add_argument('--batch_size', type=int, default = 512)
    args = parser.parse_args()

    train_dl, test_dl, bounding_box = get_dataloader(args.dataset)
    model = Nerf()
    if "ngp" in args.model.lower():
        model = NerfNGP(bounding_box = bounding_box)

    key = jax.random.key(0)
    params, key = initialize_model_variables(model, key)

    if not os.path.exists(args.save_path):
        raise ValueError(f"Could not find save path: {args.save_path}")

    generate_video(model, params, train_dl, batch_size=args.batch_size, save_path = args.save_path, model_name = args.model)
