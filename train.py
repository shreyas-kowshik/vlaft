# Minimal training script

import logging

import numpy as np
try:
    from pytorch3d.transforms import (
        euler_angles_to_matrix,
        matrix_to_euler_angles,
        matrix_to_quaternion,
        quaternion_to_matrix,
    )
except:
    print('no pytorch3d')
import torch
logger = logging.getLogger(__name__)
import functools
import math
import io
import os
import random
import re
import pickle
from multiprocessing import Value
from functools import partial
import json
from itertools import chain
from dataclasses import dataclass
import numpy as np
from PIL import Image
import copy
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Any, Callable, Dict
try:
    from petrel_client.client import Client
except:
    pass 
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import bisect
from itertools import accumulate
import copy
from typing import List
from torchvision import transforms as torchtransforms
from PIL import Image
import clip
from pdb import set_trace
import h5py
from scipy.spatial.transform import Rotation as R
import time
import flax
import optax
import jax
import jax.numpy as jnp

from utils.argument_utils import get_parser
from data.libero_seer import get_libero_pretrain_dataset
from models.bc_simple import BCSimple, generate_attention_mask, GPTConfig

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


def extract_batch(batch): # List
    # Common key guesses; tweak as needed.
    rgb_static = batch[0]
    text_tokens = batch[1]
    actions = batch[2]
    wrist_rgb = batch[3]
    states = batch[4]
    return rgb_static, text_tokens, actions, wrist_rgb, states


# TrainState class for managing model parameters and optimizer state
class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Callable = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    batch_stats: Any  # For BatchNorm running statistics
    tx: Any = nonpytree_field()
    opt_state: Any
    rng: Any  # PRNGKey for dropout etc.

    @classmethod
    def create(cls, model_def: nn.Module, params, batch_stats=None, tx=None, rng=None, **kwargs):
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1, apply_fn=model_def.apply, model_def=model_def, params=params,
            batch_stats=batch_stats, tx=tx, opt_state=opt_state, rng=rng, **kwargs,
        )

    # Call model_def.apply_fn
    def __call__(self, *args, params=None, batch_stats=None, method=None, **kwargs):
        if params is None:
            params = self.params
        if batch_stats is None:
            batch_stats = self.batch_stats
        variables = {"params": params, "batch_stats": batch_stats}
        if isinstance(method, str):
            method = getattr(self.model_def, method)
        return self.apply_fn(variables, *args, method=method, **kwargs)
    
    # Shortcut for above
    def do(self, method):
        return functools.partial(self, method=method)

    def apply_gradients(self, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        # advance step and split RNG for next call
        new_rng, _ = jax.random.split(self.rng) if self.rng is not None else (None, None)
        return self.replace(step=self.step + 1, params=new_params,
                            opt_state=new_opt_state, rng=new_rng, **kwargs)

    def apply_loss_fn(self, *, loss_fn, has_aux=False):
        """
        Takes a gradient step towards minimizing `loss_fn`.
        """
        if has_aux:
            grads, info = jax.grad(loss_fn, has_aux=has_aux)(self.params)
            return self.apply_gradients(grads=grads), info
        else:
            grads = jax.grad(loss_fn, has_aux=has_aux)(self.params)
            return self.apply_gradients(grads=grads)

# Training function
def train_step(state, images, states, actions, text_tokens, attention_mask, batch_targets):
    """Single training step using TrainState."""
    
    # fold in step to get a per-step dropout key (optional but good hygiene)
    dropout_rng = None
    if state.rng is not None:
        dropout_rng = jax.random.fold_in(state.rng, state.step)

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        # IMPORTANT: make batch_stats mutable and pass dropout rng
        (action_pred_arm, action_pred_gripper), mutable = state.apply_fn(
            variables,
            images, states, actions,
            text_tokens,
            attention_mask,
            train=True,
            mutable=['batch_stats'],
            rngs={'dropout': dropout_rng} if dropout_rng is not None else None
        )
        loss_arm = optax.l2_loss(action_pred_arm, batch_targets[:, :, :, :-1]).mean()
        loss_grip = optax.l2_loss(action_pred_gripper, batch_targets[:, :, :, -1:]).mean()
        loss = loss_arm + 0.01 * loss_grip
        # return updated batch_stats via aux
        return loss, {'loss_arm': loss_arm, 'loss_grip': loss_grip, 'batch_stats': mutable['batch_stats']}

    # Take a grad step; info carries accuracy and new batch_stats
    new_state, info = state.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
    new_state = new_state.replace(batch_stats=info['batch_stats'])
    return new_state, {'loss_arm': info['loss_arm'], 'loss_grip': info['loss_grip']}

def train_epoch(state, train_ds, args, num_batches_per_epoch=None):
    """Train for one epoch."""
    epoch_loss = 0.0
    num_batches = 0
    
    for batch in train_ds:
        rgb_static, text_tokens, actions_all, wrist_rgb, states_orig = extract_batch(batch)

        rgb_static = rgb_static[:, :-args.action_pred_steps]
        wrist_rgb = wrist_rgb[:, :-args.action_pred_steps]
        actions = actions_all[:, :-args.action_pred_steps]
        states_orig = states_orig[:, :-args.action_pred_steps]

        states = torch.cat([states_orig[..., :6], states_orig[..., [-1]]], dim=-1)
        states[..., 6:] = (states[..., 6:] + 1) // 2
        actions[..., 6:] = (actions[..., 6:] + 1) // 2

        # breakpoint()
        
        B, T, C, H, W = rgb_static.shape
        images = torch.cat([rgb_static.unsqueeze(dim=1), wrist_rgb.unsqueeze(dim=1)], dim=1)
        Ni = 2
        images = images.numpy()
        actions = actions.numpy()
        states = states.numpy()
        text_tokens = text_tokens.numpy()
        attention_mask = generate_attention_mask(T, Ni + 1 + 1, args.action_pred_steps)
        attention_mask = jnp.array(attention_mask, dtype=bool)
        # breakpoint()
        
        # Generate targets for action prediction
        # batch_targets = actions_all[:, args.action_pred_steps:]
        batch_targets = torch.cat([actions_all[:, j:args.sequence_length-args.action_pred_steps+j, :].unsqueeze(-2) for j in range(args.action_pred_steps)], dim=-2) 
        batch_targets = batch_targets.numpy()
        # breakpoint()

        # Convert all to jnp.ndarray
        images = jnp.asarray(images)
        actions = jnp.asarray(actions)
        states = jnp.asarray(states)
        text_tokens = jnp.asarray(text_tokens)
        attention_mask = jnp.asarray(attention_mask)
        batch_targets = jnp.asarray(batch_targets)
        
        # Training step
        state, train_info = train_step(state, images, states, actions, text_tokens, attention_mask, batch_targets)
        
        # Accumulate metrics
        epoch_loss += train_info['loss_arm'] + 0.01 * train_info['loss_grip']  # Using accuracy as proxy for loss tracking
        num_batches += 1
        
        # Limit batches per epoch if specified (for faster training)
        if num_batches_per_epoch and num_batches >= num_batches_per_epoch:
            break
    
    avg_loss = epoch_loss / num_batches
    return state, avg_loss

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Minimal setup to build the data loader
    clip_model, image_processor = clip.load("ViT-B/32", device=device)
    parser = get_parser(is_eval=False)
    args = parser.parse_args()
    # print(args)


    print("Building dataset...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, image_processor = clip.load("ViT-B/32", device=device)
    dataset = get_libero_pretrain_dataset(args, image_processor, clip, epoch=0, floor=False)
    loader = dataset.dataloader
    it = iter(loader)

    # Peek one batch to configure model shapes
    batch0 = next(it)
    rgb_static, text0, actions0, wrist_rgb, states_orig = extract_batch(batch0)
    states0 = torch.cat([states_orig[..., :6], states_orig[..., [-1]]], dim=-1)
    states0[..., 6:] = (states0[..., 6:] + 1) // 2
    images0 = torch.cat([rgb_static.unsqueeze(dim=1), wrist_rgb.unsqueeze(dim=1)], dim=1)
    actions0[..., 6:] = (actions0[..., 6:] + 1) // 2

    B, Ni, T, C, H, W = images0.shape
    images0 = images0.numpy()
    action_dim = actions0.shape[-1]
    state_dim = states0.shape[-1]
    actions0 = actions0.numpy()
    states0 = states0.numpy()
    text0 = text0.numpy()
    # breakpoint()

    # Build GPT config consistent with BCSimple settings
    # IMPORTANT: block_size must be >= T*(Ni + 1 + 1 + 3)
    hidden_dim = 768
    num_layers = 6
    num_heads = 8
    # action_pred_steps = 3
    gpt_conf = GPTConfig(
        block_size=T * (Ni + 1 + 1 + 3),
        num_layers=num_layers,
        num_heads=num_heads,
        num_embeds=hidden_dim,
        use_bias=True,
        dtype=None,
    )

    model_def = BCSimple(
        sequence_length=T,
        input_image_size=H,
        action_pred_steps=args.action_pred_steps,
        transformer_layers=num_layers,
        hidden_dim=hidden_dim,
        transformer_heads=num_heads,
        gripper_width=False,
        num_images=Ni,
        action_dim=action_dim,
        state_dim=state_dim,
        config=gpt_conf,
    )

    # Init model
    rng = jax.random.PRNGKey(args.seed)
    rng, params_key, dropout_key = jax.random.split(rng, 3)
    variables = model_def.init(
        {'params': params_key, 'dropout': dropout_key},
        jnp.asarray(images0), jnp.asarray(states0), jnp.asarray(actions0),
        jnp.asarray(text0),
        jnp.asarray(generate_attention_mask(T, Ni + 1 + 1, args.action_pred_steps), dtype=bool),
        train=False
    )
    params = variables['params']
    batch_stats = variables.get('batch_stats', None)

    tx = optax.adam(args.learning_rate)
    # state = TrainState.create(model_def, apply_fn=model_def.apply, params=params, tx=tx)
    state = TrainState.create(model_def, params, batch_stats=batch_stats, tx=tx, rng=params_key)
    state = state.replace(batch_stats=batch_stats, rng=rng)
    # breakpoint()

    # for data in dataset.dataloader:
    #     breakpoint()

    # Training loop
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # Train for one epoch
        state, train_loss = train_epoch(state, loader, args, num_batches_per_epoch=None)  # Limit batches for faster training
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Step: {state.step}")
        print("-" * 50)
    
    print(f"\nTraining completed!")
    print(f"Final step: {state.step}")

if __name__ == "__main__":
    main()