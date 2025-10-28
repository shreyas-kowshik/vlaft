# Minimal training script (JIT + PMAP ready)

import logging
import os
import functools
from functools import partial
from typing import Any, Callable, Dict, Optional

# 3rd-party
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
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as torchtransforms

import wandb
from omegaconf import DictConfig
import clip
from PIL import Image
from pdb import set_trace
import h5py
from scipy.spatial.transform import Rotation as R

# JAX / Flax
import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import checkpoints
from jax import lax

# Project
from utils.wandb import setup_wandb, default_wandb_config
from utils.argument_utils import get_parser
from data.libero_seer import get_libero_pretrain_dataset
from models.bc_simple import BCSimple, generate_attention_mask, GPTConfig
from utils.checkpoint import Checkpoint

logger = logging.getLogger(__name__)
nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'vlaft',
    'name': 'vlaft_libero_pretrain',
})

# ---------------------------
# Utility: batch extract
# ---------------------------
def extract_batch(batch):  # List
    rgb_static = batch[0]   # (B, T, C, H, W)
    text_tokens = batch[1]  # (B, 77) or (B, T, 77)
    actions = batch[2]      # (B, T, A)
    wrist_rgb = batch[3]    # (B, T, C, H, W)
    states = batch[4]       # (B, T, S)
    return rgb_static, text_tokens, actions, wrist_rgb, states

# ---------------------------
# Simple checkpoint helpers
# ---------------------------
def save_state(ckpt_dir: str,
               state,
               step: Optional[int] = None,
               *,
               is_replicated: bool = False,
               keep: int = 3,
               overwrite: bool = True):
    os.makedirs(ckpt_dir, exist_ok=True)
    to_save = flax.jax_utils.unreplicate(state) if is_replicated else state
    step = int(step if step is not None else getattr(to_save, "step", 0))
    if jax.process_index() == 0:
        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=to_save,
            step=step,
            keep=keep,
            overwrite=overwrite,
        )
    return step

def load_state(ckpt_dir: str,
               state_template,
               *,
               is_replicated: bool = False):
    restored = checkpoints.restore_checkpoint(ckpt_dir, target=state_template)
    return flax.jax_utils.replicate(restored) if is_replicated else restored

def log_norms_to_wandb(grads, updates, params, step):
    """Compute global L2 norms for grads/updates/params and log to Weights & Biases."""
    gnorm = optax.global_norm(grads)
    unorm  = optax.global_norm(updates)
    pnorm  = optax.global_norm(params)

    # bring to host for logging
    gnorm, unorm, pnorm = jax.device_get((gnorm, unorm, pnorm))
    wandb.log(
        {
            "grad_norm": float(gnorm),
            "update_norm": float(unorm),
            "param_norm": float(pnorm),
        },
        step=int(jax.device_get(step) if hasattr(step, "device_buffer") else int(step)),
    )
    return {"grad_norm": float(gnorm), "update_norm": float(unorm), "param_norm": float(pnorm)}

# ---------------------------
# TrainState
# ---------------------------
class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Callable = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    batch_stats: Any  # For BatchNorm running statistics (mutable collection)
    tx: Any = nonpytree_field()
    opt_state: Any
    rng: Any  # PRNGKey for dropout etc.

    @classmethod
    def create(cls, model_def, params, batch_stats=None, tx=None, rng=None, **kwargs):
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
        variables = {"params": params}
        if batch_stats is not None:
            variables["batch_stats"] = batch_stats
        if isinstance(method, str):
            method = getattr(self.model_def, method)
        return self.apply_fn(variables, *args, method=method, **kwargs)

# ---------------------------
# JIT train step (single-GPU)
# ---------------------------
@jax.jit
def train_step_jit(state, images, states, actions, text_tokens, attention_mask, batch_targets):
    # Per-step RNG management
    if state.rng is not None:
        dropout_rng, new_base_rng = jax.random.split(state.rng)
    else:
        dropout_rng, new_base_rng = None, None

    def loss_fn(params):
        variables = {"params": params}
        if state.batch_stats is not None:
            variables["batch_stats"] = state.batch_stats

        (arm_pred, grip_pred), mutable = state.apply_fn(
            variables,
            images, states, actions, text_tokens, attention_mask,
            train=True,
            mutable=['batch_stats'] if 'batch_stats' in variables else [],
            rngs={'dropout': dropout_rng} if dropout_rng is not None else None
        )
        loss_arm  = optax.l2_loss(arm_pred,  batch_targets[:, :, :, :-1]).mean()
        loss_grip = optax.l2_loss(grip_pred, batch_targets[:, :, :, -1:]).mean()
        loss = loss_arm + 0.1 * loss_grip
        new_bstats = mutable.get('batch_stats', state.batch_stats)
        return loss, (new_bstats, loss_arm, loss_grip)

    (loss, (new_bstats, loss_arm, loss_grip)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    norms_dict = log_norms_to_wandb(grads, updates, new_params, state.step)
    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        batch_stats=new_bstats,
        rng=new_base_rng,
    )
    
    return new_state, {'loss_arm': loss_arm, 'loss_grip': loss_grip, **norms_dict}

# ---------------------------
# Helpers for PMAP
# ---------------------------
def shard_batch(x):
    """Reshape leading batch for pmap: (global_b, ...) -> (n_devices, per_dev_b, ...)"""
    n_dev = jax.local_device_count()
    b = x.shape[0]
    assert b % n_dev == 0, f"Global batch {b} not divisible by n_devices={n_dev}"
    per_dev = b // n_dev
    return x.reshape(n_dev, per_dev, *x.shape[1:])

def cross_replica_mean(tree):
    """All-reduce mean across devices for any PyTree."""
    return jax.tree_util.tree_map(lambda v: lax.pmean(v, axis_name='data'), tree)

# ---------------------------
# PMAP train step (multi-GPU)
# ---------------------------
def _pmap_loss_and_grads(state, images, states, actions, text_tokens, attention_mask, batch_targets, dropout_key):
    def loss_fn(params):
        variables = {"params": params}
        if state.batch_stats is not None:
            variables["batch_stats"] = state.batch_stats

        (arm_pred, grip_pred), mutable = state.apply_fn(
            variables,
            images, states, actions, text_tokens, attention_mask,
            train=True,
            mutable=['batch_stats'] if 'batch_stats' in variables else [],
            rngs={'dropout': dropout_key}
        )
        loss_arm  = optax.l2_loss(arm_pred,  batch_targets[:, :, :, :-1]).mean()
        loss_grip = optax.l2_loss(grip_pred, batch_targets[:, :, :, -1:]).mean()
        loss = loss_arm + 0.1 * loss_grip
        # return batch_stats too
        return loss, (mutable.get('batch_stats', state.batch_stats), loss_arm, loss_grip)
    return jax.value_and_grad(loss_fn, has_aux=True)(state.params)

def _train_step_pmapped(state, images, states, actions, text_tokens, attention_mask, batch_targets):
    # Unique per-device key: fold in axis index, then split
    device_base = jax.random.fold_in(state.rng, lax.axis_index('data'))
    dropout_key, new_base_rng = jax.random.split(device_base)

    (loss, (new_bstats, loss_arm, loss_grip)), grads = _pmap_loss_and_grads(
        state, images, states, actions, text_tokens, attention_mask, batch_targets, dropout_key
    )
    # all-reduce grads + metrics
    grads     = lax.pmean(grads, axis_name='data')
    loss_arm  = lax.pmean(loss_arm, axis_name='data')
    loss_grip = lax.pmean(loss_grip, axis_name='data')

    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    # Sync BN stats across devices (if present)
    if new_bstats is not None:
        new_bstats = cross_replica_mean(new_bstats)

    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        batch_stats=new_bstats,
        rng=new_base_rng
    )
    return new_state, {'loss_arm': loss_arm, 'loss_grip': loss_grip}

train_step_pmap = jax.pmap(_train_step_pmapped, axis_name='data', donate_argnums=(0,))

# ---------------------------
# Epoch loops
# ---------------------------
def _torch_to_numpy(*xs):
    outs = []
    for x in xs:
        outs.append(x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x)
    return outs if len(outs) > 1 else outs[0]

def train_epoch_jit(state, train_ds, args, num_batches_per_epoch=None):
    epoch_loss = 0.0
    num_batches = 0

    for batch in train_ds:
        rgb_static, text_tokens, actions_all, wrist_rgb, states_orig = extract_batch(batch)

        # Trim to leave room for k-step prediction
        k = args.action_pred_steps
        rgb_static = rgb_static[:, :-k]
        wrist_rgb  = wrist_rgb[:,  :-k]
        actions    = actions_all[:, :-k]
        states_orig = states_orig[:, :-k]

        # State/Action post-processing: keep 6D arm + gripper flag (0/1)
        states = torch.cat([states_orig[..., :6], states_orig[..., [-1]]], dim=-1)
        states[..., 6:]  = (states[..., 6:] + 1) // 2
        actions[..., 6:] = (actions[..., 6:] + 1) // 2

        # Images: (B, 2, T, C, H, W)
        images = torch.cat([rgb_static.unsqueeze(1), wrist_rgb.unsqueeze(1)], dim=1)
        B, Ni, T, C, H, W = images.shape

        # Attention mask for current T
        attn = generate_attention_mask(T, Ni + 1 + 1, k)  # (L, L)
        attn = jnp.asarray(attn, dtype=bool)

        # Targets: (B, T, k, A)
        targets = torch.cat(
            [actions_all[:, j:T+j, :].unsqueeze(-2) for j in range(k)],
            dim=-2
        )

        # Torch -> NumPy -> jnp
        images_np, actions_np, states_np, tokens_np, targets_np = _torch_to_numpy(
            images, actions, states, text_tokens, targets
        )
        images_jnp      = jnp.asarray(images_np)
        actions_jnp     = jnp.asarray(actions_np)
        states_jnp      = jnp.asarray(states_np)
        text_tokens_jnp = jnp.asarray(tokens_np)
        targets_jnp     = jnp.asarray(targets_np)

        state, train_info = train_step_jit(
            state, images_jnp, states_jnp, actions_jnp, text_tokens_jnp, attn, targets_jnp
        )

        epoch_loss += float(train_info['loss_arm']) + 0.1 * float(train_info['loss_grip'])
        num_batches += 1

        if jax.process_index() == 0:
            wandb.log({
                'training/loss_arm': float(train_info['loss_arm']),
                'training/loss_grip': float(train_info['loss_grip']),
                'training/loss': float(train_info['loss_arm']) + 0.1 * float(train_info['loss_grip']),
                'training/num_batches': num_batches,
            }, step=int(state.step))

        if num_batches_per_epoch and num_batches >= num_batches_per_epoch:
            break

    return state, (epoch_loss / max(1, num_batches))

def train_epoch_pmap(state_rep, train_ds, args, num_batches_per_epoch=None):
    epoch_loss = 0.0
    num_batches = 0
    n_dev = jax.local_device_count()

    for batch in train_ds:
        rgb_static, text_tokens, actions_all, wrist_rgb, states_orig = extract_batch(batch)

        # Trim to leave room for k-step prediction
        k = args.action_pred_steps
        rgb_static = rgb_static[:, :-k]
        wrist_rgb  = wrist_rgb[:,  :-k]
        actions    = actions_all[:, :-k]
        states_orig = states_orig[:, :-k]

        # State/Action post-processing
        states = torch.cat([states_orig[..., :6], states_orig[..., [-1]]], dim=-1)
        states[..., 6:]  = (states[..., 6:] + 1) // 2
        actions[..., 6:] = (actions[..., 6:] + 1) // 2

        # Images: (B, 2, T, C, H, W)
        images = torch.cat([rgb_static.unsqueeze(1), wrist_rgb.unsqueeze(1)], dim=1)
        B, Ni, T, C, H, W = images.shape

        # Ensure global batch divisible by devices
        assert B % n_dev == 0, f"Batch {B} not divisible by n_devices={n_dev}"

        # Attention mask for current T, replicate across devices
        attn = jnp.asarray(generate_attention_mask(T, Ni + 1 + 1, k), dtype=bool)
        attn_rep = flax.jax_utils.replicate(attn)  # (n_dev, ...)

        # Targets: (B, T, k, A)
        targets = torch.cat(
            [actions_all[:, j:T+j, :].unsqueeze(-2) for j in range(k)],
            dim=-2
        )

        # Torch -> NumPy -> jnp
        images_np, actions_np, states_np, tokens_np, targets_np = _torch_to_numpy(
            images, actions, states, text_tokens, targets
        )
        images_jnp      = shard_batch(jnp.asarray(images_np))
        actions_jnp     = shard_batch(jnp.asarray(actions_np))
        states_jnp      = shard_batch(jnp.asarray(states_np))
        text_tokens_jnp = shard_batch(jnp.asarray(tokens_np))
        targets_jnp     = shard_batch(jnp.asarray(targets_np))

        state_rep, metrics_rep = train_step_pmap(
            state_rep, images_jnp, states_jnp, actions_jnp, text_tokens_jnp, attn_rep, targets_jnp
        )

        # Log (device 0)
        loss_arm  = float(jax.device_get(metrics_rep['loss_arm'][0]))
        loss_grip = float(jax.device_get(metrics_rep['loss_grip'][0]))
        epoch_loss += loss_arm + 0.1 * loss_grip
        num_batches += 1

        if jax.process_index() == 0:
            wandb.log({
                'training/loss_arm': loss_arm,
                'training/loss_grip': loss_grip,
                'training/loss': loss_arm + 0.1 * loss_grip,
                'training/num_batches': num_batches,
            }, step=int(jax.device_get(state_rep.step[0])))

        if num_batches_per_epoch and num_batches >= num_batches_per_epoch:
            break

    return state_rep, (epoch_loss / max(1, num_batches))

# ---------------------------
# Main
# ---------------------------
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
    dataset = get_libero_pretrain_dataset(args, image_processor, clip, epoch=0, floor=False)
    loader = dataset.dataloader
    it = iter(loader)

    # Peek one batch to configure shapes
    batch0 = next(it)
    rgb_static, text0, actions0, wrist_rgb, states_orig = extract_batch(batch0)
    states0 = torch.cat([states_orig[..., :6], states_orig[..., [-1]]], dim=-1)
    states0[..., 6:] = (states0[..., 6:] + 1) // 2
    images0 = torch.cat([rgb_static.unsqueeze(1), wrist_rgb.unsqueeze(1)], dim=1)
    actions0[..., 6:] = (actions0[..., 6:] + 1) // 2

    B0, Ni, T, C, H, W = images0.shape
    images0_np, actions0_np, states0_np, text0_np = _torch_to_numpy(images0, actions0, states0, text0)

    # GPT config
    hidden_dim = 768
    num_layers = 6
    num_heads = 8
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
        action_dim=actions0_np.shape[-1],
        state_dim=states0_np.shape[-1],
        config=gpt_conf,
    )

    # Init model
    rng = jax.random.PRNGKey(args.seed)
    rng, params_key, dropout_key = jax.random.split(rng, 3)
    variables = model_def.init(
        {'params': params_key, 'dropout': dropout_key},
        jnp.asarray(images0_np), jnp.asarray(states0_np), jnp.asarray(actions0_np),
        jnp.asarray(text0_np),
        jnp.asarray(generate_attention_mask(T, Ni + 1 + 1, args.action_pred_steps), dtype=bool),
        train=False
    )
    params = variables['params']
    batch_stats = variables.get('batch_stats', None)

    tx = optax.adam(args.learning_rate)
    state = TrainState.create(model_def, params, batch_stats=batch_stats, tx=tx, rng=rng)

    # Decide mode: JIT vs PMAP
    use_pmap = jax.local_device_count() > 1
    print(f"local_device_count={jax.local_device_count()} -> {'PMAP' if use_pmap else 'JIT'} mode")

    # Build fresh loader (rewound)
    loader = dataset.dataloader

    if use_pmap:
        # Replicate state
        state_rep = flax.jax_utils.replicate(state)
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            state_rep, train_loss = train_epoch_pmap(state_rep, loader, args, num_batches_per_epoch=None)
            print(f"  Train Loss: {train_loss:.4f}")
            print("-" * 50)

            # Save replicated state
            ckpt_dir = os.path.join(args.root_dir, 'checkpoints', 'trial')
            save_state(ckpt_dir, state_rep, step=int(jax.device_get(state_rep.step[0])), is_replicated=True)
        print("Training completed.")
        run.finish()
    else:
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            state, train_loss = train_epoch_jit(state, loader, args, num_batches_per_epoch=None)
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Step: {state.step}")
            print("-" * 50)

            ckpt_dir = os.path.join(args.root_dir, 'checkpoints', 'trial')
            save_state(ckpt_dir, state, step=int(state.step), is_replicated=False)

        print("Training completed.")
        print(f"Final step: {state.step}")
        run.finish()

if __name__ == "__main__":
    main()
