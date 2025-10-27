#!/usr/bin/env python3
import os
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training import train_state as flax_train_state

# --- your code / dataset bits ---
import torch
import clip  # used by your dataset factory
from data.libero_seer import get_libero_pretrain_dataset

# --- your model ---
from bc_simple import BCSimple, generate_attention_mask, GPTConfig


# ---------------------------
# Utilities
# ---------------------------

def to_numpy(x):
    if isinstance(x, jnp.ndarray):
        return np.asarray(x)
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def maybe_nhwc(images_np):
    """
    Accepts either:
      (B, num_images, T, H, W, C)  -> already NHWC (returns as-is), or
      (B, num_images, T, C, H, W)  -> converts to NHWC.
    """
    if images_np.ndim != 6:
        raise ValueError(f"images must be rank-6 (B,Ni,T,H,W,C) or (B,Ni,T,C,H,W); got {images_np.shape}")
    B, Ni, T = images_np.shape[:3]
    last = images_np.shape[-1]
    if last in (1, 3, 4):
        # already NHWC
        return images_np
    # assume NCHW: (B,Ni,T,C,H,W) -> (B,Ni,T,H,W,C)
    return np.transpose(images_np, (0, 1, 2, 4, 5, 3))


def ensure_int32(x):
    x_np = to_numpy(x)
    return x_np.astype(np.int32)


def build_future_targets(actions_btd: jnp.ndarray, k: int) -> jnp.ndarray:
    """
    Build future target windows of length k for each time step:
      actions: (B, T, D) -> (B, T, k, D)
    For t where t+s exceeds T-1, we pad by repeating the last action.
    """
    B, T, D = actions_btd.shape
    pad = jnp.repeat(actions_btd[:, -1:, :], k - 1, axis=1)          # (B, k-1, D)
    padded = jnp.concatenate([actions_btd, pad], axis=1)             # (B, T+k-1, D)

    def slice_s(s):
        # take length-T slice starting at s along time
        return jax.lax.dynamic_slice_in_dim(padded, s, T, axis=1)    # (B, T, D)

    futures = jnp.stack([slice_s(s) for s in range(k)], axis=2)      # (B, T, k, D)
    return futures


def extract_batch(batch: Dict[str, Any]):
    """
    Try to extract the fields your model needs from a batch produced by get_libero_pretrain_dataset.
    Adjust the key names here if your dataloader returns different ones.
    Expected shapes (after conversion below):
      images: (B, num_images, T, H, W, C), float32
      states: (B, T, state_dim), float32
      actions:(B, T, action_dim), float32
      text_tokens: (B, T, 77), int32
    """
    # Common key guesses; tweak as needed.
    possible_image_keys = ['images', 'rgb', 'image']
    possible_state_keys = ['states', 'proprio', 'state']
    possible_action_keys = ['actions', 'action']
    possible_text_keys = ['text_tokens', 'clip_tokens', 'input_ids', 'text_ids']

    def pick(keys):
        for k in keys:
            if k in batch:
                return k
        raise KeyError(f"None of the keys {keys} found in batch. Available: {list(batch.keys())}")

    k_img = pick(possible_image_keys)
    k_act = pick(possible_action_keys)

    # states may or may not be present; default zeros if missing.
    k_state = None
    try:
        k_state = pick(possible_state_keys)
    except KeyError:
        pass

    # tokens can be a ready-made token tensor or raw ids; ensure shape (B,T,77) and int32
    k_txt = pick(possible_text_keys)

    images = maybe_nhwc(to_numpy(batch[k_img])).astype(np.float32)        # (B,Ni,T,H,W,C)
    actions = to_numpy(batch[k_act]).astype(np.float32)                    # (B,T,action_dim)
    if k_state is not None:
        states = to_numpy(batch[k_state]).astype(np.float32)               # (B,T,state_dim)
    else:
        B, T = actions.shape[0], actions.shape[1]
        states = np.zeros((B, T, 7), dtype=np.float32)                     # fallback

    text_tokens = ensure_int32(batch[k_txt])                               # (B,T,77) expected

    return images, states, actions, text_tokens


# ---------------------------
# Train state
# ---------------------------

class TrainState(flax_train_state.TrainState):
    batch_stats: Any = None
    rng: Any = None  # keep a dropout/base rng


# ---------------------------
# Loss / step functions
# ---------------------------

def make_loss_fn(model_def: BCSimple, num_images: int, action_pred_steps: int):
    """
    Returns loss_fn bound to a specific model and its static config.
    """
    def loss_fn(params, batch_stats, rng, batch):
        images, states, actions, text_tokens = batch
        B, Ni, T, H, W, C = images.shape

        # Build the block size and mask (L = T * (Ni + 1 + 1 + 3))
        L = T * (num_images + 1 + 1 + 3)
        attention_mask = generate_attention_mask(T, num_images + 1 + 1, action_pred_steps)
        attention_mask = jnp.asarray(attention_mask, dtype=bool)  # (L, L)

        variables = {'params': params}
        if batch_stats is not None:
            variables['batch_stats'] = batch_stats

        # Forward
        (pred_arm, pred_grip), updates = model_def.apply(
            variables,
            jnp.asarray(images),
            jnp.asarray(states),
            jnp.asarray(actions),
            jnp.asarray(text_tokens),
            attention_mask,
            train=True,
            rngs={'dropout': rng},
            mutable=['batch_stats'] if ('batch_stats' in variables) else []
        )

        new_batch_stats = updates.get('batch_stats', batch_stats)

        # Targets for k-step prediction: (B, T, k, D)
        targets_btkd = build_future_targets(jnp.asarray(actions), action_pred_steps)

        arm_targets = targets_btkd[:, :, :, :-1]   # (B,T,k,action_dim-1)
        grip_targets = targets_btkd[:, :, :, -1:]  # (B,T,k,1)

        # MSE on arm + gripper
        loss_arm = jnp.mean((pred_arm - arm_targets) ** 2)
        loss_grip = jnp.mean((pred_grip - grip_targets) ** 2)
        loss = loss_arm + loss_grip

        return loss, (new_batch_stats, {'loss': loss, 'arm_mse': loss_arm, 'grip_mse': loss_grip})
    return loss_fn


@jax.jit
def train_step(state: TrainState, batch, loss_fn):
    rng, step_key = jax.random.split(state.rng)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_batch_stats, scalars)), grads = grad_fn(state.params, state.batch_stats, step_key, batch)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_state = state.replace(
        params=new_params, opt_state=new_opt_state,
        batch_stats=new_batch_stats, rng=rng
    )
    return new_state, scalars


@jax.jit
def eval_step(state: TrainState, batch, model_def: BCSimple, num_images: int, action_pred_steps: int):
    images, states, actions, text_tokens = batch
    B, Ni, T, H, W, C = images.shape

    attention_mask = generate_attention_mask(T, num_images + 1 + 1, action_pred_steps)
    attention_mask = jnp.asarray(attention_mask, dtype=bool)

    variables = {'params': state.params}
    if state.batch_stats is not None:
        variables['batch_stats'] = state.batch_stats

    pred_arm, pred_grip = model_def.apply(
        variables,
        jnp.asarray(images),
        jnp.asarray(states),
        jnp.asarray(actions),
        jnp.asarray(text_tokens),
        attention_mask,
        train=False,  # deterministic
    )

    targets_btkd = build_future_targets(jnp.asarray(actions), action_pred_steps)
    arm_targets = targets_btkd[:, :, :, :-1]
    grip_targets = targets_btkd[:, :, :, -1:]
    arm_mse = jnp.mean((pred_arm - arm_targets) ** 2)
    grip_mse = jnp.mean((pred_grip - grip_targets) ** 2)
    return {'loss': arm_mse + grip_mse, 'arm_mse': arm_mse, 'grip_mse': grip_mse}


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    # dataset args: reuse your original parser so the dataloader builds correctly
    from utils.argument_utils import get_parser as get_data_parser
    data_parser = get_data_parser(is_eval=False)
    # Merge: first parse our args, then the rest go to dataset
    args, unknown = parser.parse_known_args()
    data_args = data_parser.parse_args(unknown)

    print("Building dataset...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, image_processor = clip.load("ViT-B/32", device=device)
    dataset = get_libero_pretrain_dataset(data_args, image_processor, clip, epoch=0, floor=False)
    loader = dataset.dataloader
    it = iter(loader)

    # Peek one batch to configure model shapes
    batch0 = next(it)
    images0, states0, actions0, text0 = extract_batch(batch0)

    B, Ni, T, H, W, C = images0.shape
    action_dim = actions0.shape[-1]
    state_dim = states0.shape[-1]

    # Build GPT config consistent with BCSimple settings
    # IMPORTANT: block_size must be >= T*(Ni + 1 + 1 + 3)
    hidden_dim = 768
    num_layers = 6
    num_heads = 8
    action_pred_steps = 3
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
        action_pred_steps=action_pred_steps,
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
        jnp.asarray(generate_attention_mask(T, Ni + 1 + 1, action_pred_steps), dtype=bool),
        train=False
    )
    params = variables['params']
    batch_stats = variables.get('batch_stats', None)

    tx = optax.adam(args.lr)
    state = TrainState.create(apply_fn=model_def.apply, params=params, tx=tx)
    state = state.replace(batch_stats=batch_stats, rng=rng)

    # Build loss fn bound to current model/static config
    loss_fn = make_loss_fn(model_def, num_images=Ni, action_pred_steps=action_pred_steps)

    print("Start training...")
    step = 0
    running = {'loss': 0.0, 'arm_mse': 0.0, 'grip_mse': 0.0}
    while step < args.max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        images, states, actions, text_tokens = extract_batch(batch)

        # Convert to jnp once here
        batch_jnp = (
            jnp.asarray(images), jnp.asarray(states),
            jnp.asarray(actions), jnp.asarray(text_tokens)
        )

        state, scalars = train_step(state, batch_jnp, loss_fn)

        # logging
        for k in running:
            running[k] += float(scalars[k])
        step += 1
        if step % args.log_every == 0:
            mean_vals = {k: v/args.log_every for k, v in running.items()}
            print(f"[step {step}] loss={mean_vals['loss']:.6f}  arm_mse={mean_vals['arm_mse']:.6f}  grip_mse={mean_vals['grip_mse']:.6f}")
            running = {k: 0.0 for k in running}

    print("Done.")


if __name__ == "__main__":
    main()
