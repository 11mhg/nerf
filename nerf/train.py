import os, jax
import flax 
import math
import optax
import orbax
import wandb
import functools
import numpy as np
from jax import lax
import orbax.checkpoint
import jax.numpy as jnp
import flax.linen as nn
from flax.training import orbax_utils
from flax.training.train_state import TrainState

from prefetch_generator import BackgroundGenerator

from nerf.dataloader import Nerf_Data, get_dataloader
from nerf.models.utils import get_checkpoint_manager, calculate_alphas, calculate_accumulated_transformation
from nerf.models.base import ( 
    Nerf, initialize_model_variables
)
from nerf.models.ngp import (
    NerfNGP
)
from jax import config
config.update("jax_debug_nans", True)

def render(model: Nerf, params: dict, position: jnp.array, direction: jnp.array, t_vals: jnp.array):
    """
    Compute the pixel colors of a camera looking at sample positions and directions using a nerf model.

    Args:
        model (Nerf): The nerf model to use for rendering.
        params: The parameters of the nerf model.
        position (jnp.array): The sample positions in [bs, num_samples, 3].
        direction (jnp.array): The sample directions in [bs, num_samples, 3].
        t_vals (jnp.array): The ray distance values.

    Returns:
        tuple: A tuple containing the computed rgb values, depths, sigma values, and weights.
    """

    rgb, sigma = model.apply(
        params,
        position = position,
        direction = direction
    )
    rgb = rgb.reshape(position.shape)
    
    deltas = jnp.reshape( t_vals, (-1, t_vals.shape[-1]) )
    delta_inf = jnp.array(
        [[1e10]],
    ).repeat(deltas.shape[0], axis = 0)
    deltas = jnp.concatenate(
        (deltas[...,1:] - deltas[...,:-1], delta_inf),
        axis = -1
    )

    alphas = calculate_alphas( sigma[...,0], deltas )
    weights = calculate_accumulated_transformation(alphas)[...,None] * alphas[...,None]

    colours = (rgb * weights).sum(axis = 1)
    depths = jnp.sum(weights[...,0] * t_vals, axis = 1)

    weight_sum = weights.sum(axis = -1).sum(axis = -1)

    colours = (colours + 1) - weight_sum[...,None] # (regularize for white background)

    return colours, depths, sigma, weights

def train_step(train_state: TrainState, sample: dict):

    def loss_fn(params):
        #image: jnp.array, position: jnp.array, direction: jnp.array, t_vals: jnp.array
        colours, depths, sigma, weights = train_state.apply_fn(
            params = params,
            position = sample['position'],
            direction = sample['direction'],
            t_vals = sample['t_vals']
        )
        return jnp.mean((sample['image'] - colours)**2)
    
    l, grads = jax.value_and_grad(loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads = grads)
    return l, train_state

def test_step(train_state: TrainState, sample: dict):
    colours, depths, sigma, weights = train_state.apply_fn(
        params = train_state.params,
        position = sample['position'],
        direction = sample['direction'],
        t_vals = sample['t_vals']
    )
    l = jnp.mean((sample['image'] - colours)**2)
    return colours, l

def train( 
        model: Nerf, 
        params: dict, 
        train_loader: Nerf_Data, 
        test_loader: Nerf_Data, 
        batch_size: int = 512,
        num_epochs: int = 15,
        save_path: str = './model-ckpt',
        max_to_keep: int = 2,
        model_name: str = 'nerf',
    ):
    checkpoint_manager = get_checkpoint_manager(
        save_path, max_to_keep = max_to_keep, create = True
    ) 
   
    transition_steps = ((100*100*100)//batch_size) * (num_epochs // 2)
    schedule = model.get_learning_rate_schedule() 

    optimizer = model.get_optimizer(schedule) 

    state = TrainState.create(
        apply_fn = functools.partial( render, model = model ),
        params = params,
        tx = optimizer
    )

    save_args = orbax_utils.save_args_from_target({'state': state, 'best_loss': np.Inf})

    def save_fn(step, state, best_loss = np.Inf):
        ckpt = {
            'state': state,
            'best_loss': best_loss
        }
        checkpoint_manager.save(step, ckpt, save_kwargs={'save_args':save_args})
        return
    
    def load_fn(step, state, best_loss = np.Inf):
        target = {'state': state, 'best_loss': best_loss}
        ckpt = checkpoint_manager.restore(step, items=target )
        state = ckpt['state']
        best_loss = ckpt['best_loss']
        return state, best_loss


    best_loss = np.Inf
    if checkpoint_manager.latest_step() is not None: 
        print(f"RESTORING FROM CHECKPOINT {checkpoint_manager.latest_step()}")
        state, best_loss = load_fn(checkpoint_manager.latest_step(), state, best_loss)
    else:
        print(f"INITIALIZING FROM SCRATCH")
        save_fn(0, state, best_loss)

    run = wandb.init(
        project="nerf",
        name=f"{model_name}",
        config = {
            'batch_size': batch_size,
            'num_epochs': num_epochs
        }
    )

    total_train_step_count = 0
    total_test_step_count = 0
    for epoch in range(num_epochs):
        epoch_train_loss = []
        for ind, sample in BackgroundGenerator(enumerate(train_loader), max_prefetch = 100):
            num_batches = math.ceil(sample['image'].shape[0] / batch_size)
            for batch_ind in range(num_batches):
                batch_sample = {
                    'image': sample['image'][batch_ind*batch_size:(batch_ind+1)*batch_size],
                    'position': sample['position'][batch_ind*batch_size:(batch_ind+1)*batch_size],
                    'direction': sample['direction'][batch_ind*batch_size:(batch_ind+1)*batch_size],
                    't_vals': sample['t_vals'][batch_ind*batch_size:(batch_ind+1)*batch_size],
                }
                batch_sample['direction'] = batch_sample['direction'][:,None,:].repeat(
                    batch_sample['position'].shape[1],
                    axis = 1
                )
                l, state = train_step(state, batch_sample)
                psnr = 10 * jnp.log10(1./l)
                wandb.log({
                    "train_loss": l, 
                    "train_psnr": psnr,
                    "learning_rate": schedule(total_train_step_count)
                }, step = state.step)
                epoch_train_loss.append(float(l))
                total_train_step_count += 1
                if total_train_step_count % 10 == 0:
                    print(f"Epoch: {epoch} train_step: {total_train_step_count} Loss: {np.mean(epoch_train_loss)}")
        
        epoch_test_loss = []
        image = None
        for ind, sample in enumerate(test_loader):
            generated_image = []
            num_batches = math.ceil(sample['image'].shape[0] / batch_size)
            for batch_ind in range(num_batches):
                batch_sample = {
                    'image': sample['image'][batch_ind*batch_size:(batch_ind+1)*batch_size],
                    'position': sample['position'][batch_ind*batch_size:(batch_ind+1)*batch_size],
                    'direction': sample['direction'][batch_ind*batch_size:(batch_ind+1)*batch_size],
                    't_vals': sample['t_vals'][batch_ind*batch_size:(batch_ind+1)*batch_size],
                }
                batch_sample['direction'] = batch_sample['direction'][:,None,:].repeat(
                    batch_sample['position'].shape[1],
                    axis = 1
                )
                colours, l = test_step(state, batch_sample)
                generated_image.append(colours)
                epoch_test_loss.append(float(l))
                total_test_step_count += 1
            generated_image = jnp.concatenate(generated_image, axis = 0)
            generated_image = jnp.asarray(generated_image)
            generated_image = np.array(generated_image)
            generated_image = np.reshape(
                generated_image,
                (100,100,3)
            )
            image = wandb.Image(
                generated_image
            )

        wandb.log({
            "test_image": image,
            "test_loss": np.mean(epoch_test_loss),
            "test_psnr": 10 * jnp.log10(1./np.mean(epoch_test_loss))
        }, step = state.step)

        if np.mean(epoch_test_loss) < best_loss:
            best_loss = np.mean(epoch_test_loss)
            save_fn(total_train_step_count, state, best_loss)
            print(f"Model Saved {total_train_step_count}")

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog = "TrainNerf",
        description = "Train a nerf model"
    )
    parser.add_argument('--dataset', '-d', type=str, help="The dataset to use for training the nerf model.",
        default = "./tiny_nerf_data.npz"
    )
    parser.add_argument('--model', '-m', type=str, help="The model to use: nerf or ngp. Defaults to nerf.",
        default = "nerf"
    )
    parser.add_argument('--save-path', '-s', type=str, help = "The save path for the model checkpoints.",
        default = "./model-ckpt"
    )
    parser.add_argument('--batch_size', type=int, help = "The batch size for training.", default=512)
    args = parser.parse_args()

    train_dl, test_dl, bounding_box = get_dataloader(args.dataset)

    model = Nerf()
    if "ngp" in args.model.lower():
        model = NerfNGP(bounding_box = bounding_box)

    key = jax.random.key(0)

    params, key = initialize_model_variables(model, key)

    os.makedirs(args.save_path,exist_ok = True)
    train(model, params, train_dl, test_dl, batch_size=args.batch_size, save_path = args.save_path, model_name = args.model)



