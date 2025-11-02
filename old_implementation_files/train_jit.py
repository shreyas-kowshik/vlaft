# Minimal training script (JIT-only)

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
from typing import Any, Callable, Dict, Optional
import wandb
from utils.wandb import setup_wandb, default_wandb_config
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
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    _tz = ZoneInfo("America/New_York")
except Exception:
    _tz = None
from flax.training import checkpoints

from utils.argument_utils import get_parser
from data.libero_seer import get_libero_pretrain_dataset, get_libero_finetune_dataset
from models.bc_simple import BCSimple, generate_attention_mask, GPTConfig
from utils.checkpoint import Checkpoint

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'vlaft',
    'name': 'vlaft_libero_pretrain',
})

def extract_batch(batch):  # List
    rgb_static = batch[0]
    text_tokens = batch[1]
    actions = batch[2]
    wrist_rgb = batch[3]
    states = batch[4]
    return rgb_static, text_tokens, actions, wrist_rgb, states

def save_state(ckpt_dir: str,
               state,
               step: int,
               *,
               prefix: str = 'checkpoint_',
               keep: int = 0,                 # keep=0 ⇒ keep everything
               overwrite: bool = False):
    """
    Save a Flax/JAX PyTree (e.g., TrainState) to `ckpt_dir` as a unique file.
    Files are named like `{prefix}{step}` (e.g., epoch_0001).

    - `prefix` controls the filename prefix (e.g., 'epoch_').
    - `keep=0` keeps all checkpoints (no deletion policy).
    - `overwrite=False` prevents clobbering an existing step file.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir,
        target=state,
        step=int(step),
        prefix=prefix,
        keep=keep,
        overwrite=overwrite,
    )
    return step

def load_state(ckpt_dir: str,
               state_template,
               *,
               is_replicated: bool = False):
    """Simple restore (no replication in JIT-only mode)."""
    restored = checkpoints.restore_checkpoint(ckpt_dir, target=state_template)
    return restored

# ---------------------------
# TrainState (no pmap usage)
# ---------------------------
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
        opt_state = tx.init(params) if tx is not None else None
        return cls(
            step=1, apply_fn=model_def.apply, model_def=model_def, params=params,
            batch_stats=batch_stats, tx=tx, opt_state=opt_state, rng=rng, **kwargs,
        )

    # Call model_def.apply
    def __call__(self, *args, params=None, batch_stats=None, method=None, **kwargs):
        if params is None:
            params = self.params
        if batch_stats is None:
            batch_stats = self.batch_stats
        variables = {"params": params, "batch_stats": batch_stats}
        if isinstance(method, str):
            method = getattr(self.model_def, method)
        return self.apply_fn(variables, *args, method=method, **kwargs)

# ---------------------------
# JIT-compiled train step
# ---------------------------
@jax.jit
def train_step(state: TrainState,
               images: jnp.ndarray,
               states: jnp.ndarray,
               actions: jnp.ndarray,
               text_tokens: jnp.ndarray,
               attention_mask: jnp.ndarray,
               batch_targets: jnp.ndarray):
    """
    Single-device, JIT-compiled training step.
    Returns new_state and scalar metrics (including norms).
    """

    # Per-step RNG for dropout (fold in step for deterministic variation)
    if state.rng is not None:
        step_key = jax.random.fold_in(state.rng, state.step)
        dropout_rng, new_base_rng = jax.random.split(step_key)
    else:
        dropout_rng, new_base_rng = None, None

    def loss_fn(params, batch_stats):
        variables = {"params": params, "batch_stats": batch_stats}
        (action_pred_arm, action_pred_gripper), mutable = state.apply_fn(
            variables,
            images, states, actions,
            text_tokens,
            attention_mask,
            train=True,
            mutable=['batch_stats'],
            rngs={'dropout': dropout_rng} if dropout_rng is not None else None
        )
        loss_arm = optax.huber_loss(action_pred_arm, batch_targets[:, :, :, :-1]).mean()
        loss_grip = optax.huber_loss(action_pred_gripper, batch_targets[:, :, :, -1:]).mean()
        loss = loss_arm + 0.1 * loss_grip
        new_bstats = mutable['batch_stats']
        return loss, (new_bstats, loss_arm, loss_grip)

    (loss, (new_bstats, loss_arm, loss_grip)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, state.batch_stats
    )

    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    # Norms for logging
    grad_norm   = optax.global_norm(grads)
    update_norm = optax.global_norm(updates)
    param_norm  = optax.global_norm(new_params)

    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        batch_stats=new_bstats,
        rng=new_base_rng,
    )
    metrics = {
        'loss_arm': loss_arm,
        'loss_grip': loss_grip,
        'grad_norm': grad_norm,
        'update_norm': update_norm,
        'param_norm': param_norm,
    }
    return new_state, metrics

# ---------------------------
# Epoch loop (host logging)
# ---------------------------
def train_epoch(state, train_ds, args, num_batches_per_epoch=None, epoch=0, lr_schedule=None, ckpt_dir=None):
    """Train for one epoch (JIT-only)."""
    epoch_loss = 0.0
    num_batches = 0

    # Optional: save a checkpoint at start of epoch via your custom Checkpoint util
    # ckpt_dir = os.path.join(args.root_dir, 'checkpoints', 'trial')
    # if jax.process_index() == 0:
    #     model_single = state
    #     cp = Checkpoint(ckpt_dir, parallel=False)
    #     cp.set_model(model_single)
    #     cp.save()
    #     del cp, model_single
    #     print(f"Checkpoint saved at {ckpt_dir}")
    # Save a checkpoint each epoch
    if ckpt_dir is None:
        ckpt_dir = os.path.join(args.root_dir, "checkpoints", f"jit_default")
    
    if jax.process_index() == 0:
        # save_state(ckpt_dir, state, step=int(state.step), is_replicated=False)
        save_state(
            ckpt_dir,
            state,
            step=epoch + 1,
            prefix='epoch_',              # => epoch_1, epoch_2, ...
        )

    for batch in train_ds:
        rgb_static, text_tokens, actions_all, wrist_rgb, states_orig = extract_batch(batch)

        actions_all[..., 6:] = (actions_all[..., 6:] + 1) // 2

        # Leave room for k-step prediction targets
        k = args.action_pred_steps
        rgb_static = rgb_static[:, :-k]
        wrist_rgb  = wrist_rgb[:,  :-k]
        actions    = actions_all[:, :-k]
        states_orig = states_orig[:, :-k]

        # Keep 6D arm + gripper flag, map gripper from [-1,1] -> {0,1}
        states = torch.cat([states_orig[..., :6], states_orig[..., [-1]]], dim=-1)
        states[..., 6:]  = (states[..., 6:] + 1) // 2

        # Build images (B, 2, T, C, H, W)
        images = torch.cat([rgb_static.unsqueeze(1), wrist_rgb.unsqueeze(1)], dim=1)
        B, Ni, T, C, H, W = images.shape

        # Attention mask (L, L) -> jnp.bool_
        attention_mask = generate_attention_mask(T, Ni + 1 + 1, k)
        attention_mask = jnp.asarray(attention_mask, dtype=bool)

        # Targets (B, T, k, A)
        targets = torch.cat(
            [actions_all[:, j:args.sequence_length - k + j, :].unsqueeze(-2) for j in range(k)],
            dim=-2
        )

        # Torch -> NumPy -> JAX
        def to_np(*xs):
            outs = []
            for x in xs:
                outs.append(x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x)
            return outs if len(outs) > 1 else outs[0]

        images_np, actions_np, states_np, tokens_np, targets_np = to_np(
            images, actions, states, text_tokens, targets
        )
        images_jnp      = jnp.asarray(images_np)
        actions_jnp     = jnp.asarray(actions_np)
        states_jnp      = jnp.asarray(states_np)
        text_tokens_jnp = jnp.asarray(tokens_np)
        targets_jnp     = jnp.asarray(targets_np)

        # JIT step
        state, train_info = train_step(
            state, images_jnp, states_jnp, actions_jnp, text_tokens_jnp, attention_mask, targets_jnp
        )

        # Host logging (convert DeviceArrays to Python floats)
        info_host = jax.device_get(train_info)
        epoch_loss += float(info_host['loss_arm']) + 0.1 * float(info_host['loss_grip'])
        num_batches += 1

        # Save every `k` steps here
        if state.step % args.save_every_iter == 0:
            save_state(
                ckpt_dir,
                state,
                step=int(state.step),
                prefix='epoch_nb_' + str(num_batches),
            )
            print(f"Checkpoint saved at {ckpt_dir}")

        # Current learning rate (host value) for logging
        lr_val = None
        if lr_schedule is not None:
            lr_val = float(jax.device_get(lr_schedule(int(state.step))))

        if jax.process_index() == 0:
            wandb.log({
                'training/loss_arm': float(info_host['loss_arm']),
                'training/loss_grip': float(info_host['loss_grip']),
                'training/loss': float(info_host['loss_arm']) + 0.1 * float(info_host['loss_grip']),
                'training/grad_norm': float(info_host['grad_norm']),
                'training/update_norm': float(info_host['update_norm']),
                'training/lr': lr_val if lr_val is not None else None,
                'training/param_norm': float(info_host['param_norm']),
                'training/num_batches': num_batches,
            }, step=int(state.step))

        if num_batches_per_epoch and num_batches >= num_batches_per_epoch:
            break

    avg_loss = epoch_loss / max(1, num_batches)
    return state, avg_loss

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, image_processor = clip.load("ViT-B/32", device=device)
    parser = get_parser(is_eval=False)
    args = parser.parse_args()

    run = wandb.init(
        project="vlaft",
        name="bc_run_001",
        config={
            "lr": args.learning_rate,
            "batch_size": args.batch_size,
            "seed": 0,
            "dataset": "LIBERO",
            "model": "BCSimple",
        },
    )

    print("Building dataset...")
    # dataset = get_libero_pretrain_dataset(args, image_processor, clip, epoch=0, floor=False)
    dataset = get_libero_finetune_dataset(args, image_processor, clip, epoch=0, floor=False, dset_name="libero_10_converted_kitchen_scene6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it")
    loader = dataset.dataloader
    it = iter(loader)

    # Peek one batch to configure shapes
    batch0 = next(it)
    rgb_static, text0, actions0, wrist_rgb, states_orig = extract_batch(batch0)
    # Assertions for sanity checking
    assert len(rgb_static.shape) == 5
    assert len(wrist_rgb.shape) == 5
    assert len(states_orig.shape) == 3
    assert len(actions0.shape) == 3
    assert len(text0.shape) == 2
    assert rgb_static.shape[0] == wrist_rgb.shape[0] == states_orig.shape[0] == actions0.shape[0] == text0.shape[0]
    assert rgb_static.shape[1] == wrist_rgb.shape[1] == states_orig.shape[1] == actions0.shape[1]

    states0 = torch.cat([states_orig[..., :6], states_orig[..., [-1]]], dim=-1)
    states0[..., 6:] = (states0[..., 6:] + 1) // 2
    images0 = torch.cat([rgb_static.unsqueeze(1), wrist_rgb.unsqueeze(1)], dim=1)
    actions0[..., 6:] = (actions0[..., 6:] + 1) // 2

    B, Ni, T, C, H, W = images0.shape
    images0 = images0.numpy()
    action_dim = actions0.shape[-1]
    state_dim = states0.shape[-1]
    actions0 = actions0.numpy()
    states0 = states0.numpy()
    text0 = text0.numpy()

    # GPT config (block_size must be >= T*(Ni + 1 + 1 + 3))
    hidden_dim = 768
    num_layers = 12
    num_heads = 12
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

    # Init model (JIT-only)
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

    # Get param count
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"\n\n\n\n\nParameter count: {param_count}\n\n\n\n\n")
    breakpoint()

    # tx = optax.adam(args.learning_rate)  
    # -------------------------------
    # Cosine LR schedule w/ warmup
    # -------------------------------
    # Total steps is an estimate; if your dataloader has a known length, replace with
    #   total_steps = args.num_epochs * steps_per_epoch
    total_steps = getattr(args, 'max_steps', args.num_epochs * len(loader))
    warmup_steps = max(1, int(0.01 * total_steps))  # 1% warmup (adjust to match Seer config)
    decay_steps  = max(1, total_steps - warmup_steps)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,                 # start at 0
        peak_value=float(args.learning_rate),
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=0.0                   # decay to 0
    )
    # Use schedule in Adam optimizer
    tx = optax.adam(lr_schedule)

    state = TrainState.create(model_def, params, batch_stats=batch_stats, tx=tx, rng=rng)

    # Load checkpoint if provided
    # if args.resume_from_checkpoint is not None:
    #     state = checkpoints.restore_checkpoint(args.resume_from_checkpoint, target=state)
    #     print(f"Checkpoint loaded from {args.resume_from_checkpoint}")

    #     # Make some changes to ensure recompilation does not happed at every step
    #     state = jax.tree_util.tree_map(lambda x: jnp.asarray(x) if isinstance(x, (np.ndarray, np.generic)) else x, state)
    #     state = jax.device_put(state)
    #     state = state.replace(step=jnp.asarray(state.step))

    run_stamp = (datetime.now(_tz) if _tz else datetime.now()).strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(args.root_dir, "checkpoints", f"jit_{run_stamp}")

    # Training loop (JIT-only)
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        # state, train_loss = train_epoch(state, loader, args, num_batches_per_epoch=None, epoch=epoch)
        state, train_loss = train_epoch(state, loader, args, num_batches_per_epoch=None, epoch=epoch, lr_schedule=lr_schedule, ckpt_dir=ckpt_dir)
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Step: {state.step}")
        print("-" * 50)

    print("\nTraining completed!")
    print(f"Final step: {state.step}")
    run.finish()

if __name__ == "__main__":
    main()