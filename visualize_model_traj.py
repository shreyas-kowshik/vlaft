# main.py
from re import L
import tensorflow as tf
import os
import shutil
# CRITICAL: Disable GPU for TensorFlow to prevent GPU memory allocation during dataset creation
# TensorFlow will allocate GPU memory by default if it detects a GPU, even for dataset operations
tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
tf.random.set_seed(0)

import jax.numpy as jnp
import jax
import torch
import numpy as np

import rlds
import clip # To tokenize

import optax
import flax
import wandb
import tqdm
import functools
from flax.training import checkpoints

from data.libero_rlds import episode_to_windows_with_prefix
from models.bc_simple import generate_attention_mask, BCSimple, GPTConfig
from eval_libero_clean import eval_libero10, save_rgbs_to_gif

def main():
    # CONSTANTS #
    CKPT_DIR = "/data/user_data/skowshik/checkpoints/libero_10_converted_kitchen_scene6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it/ckpt_dir/checkpoint_29700"
    seed = 0
    image_primary_size = 224
    image_wrist_size = 224
    window_size = 13 # Actual history length is window_size - action_pred_steps
    NUM_IMAGES = 2 # Wrist + Static Camera
    action_pred_steps = 3
    history_length = window_size - action_pred_steps
    action_dim = 7
    state_dim = 7
    gripper_width = True
    if gripper_width:
        state_dim = 8
    # Model #
    hidden_dim = 768
    num_layers = 24
    num_heads = 12
    dropout_rate = 0.0
    
    # MODEL CREATION #
    gpt_conf = GPTConfig(
        block_size=(history_length + action_pred_steps) * (NUM_IMAGES + 1 + 1 + 3),
        num_layers=num_layers,
        num_heads=num_heads,
        num_embeds=hidden_dim,
        use_bias=True,
        dtype=jnp.float32,
        dropout_rate=dropout_rate,
    )

    model_def = BCSimple(
        sequence_length=history_length,
        input_image_size=image_primary_size,
        action_pred_steps=action_pred_steps,
        transformer_layers=num_layers,
        hidden_dim=hidden_dim,
        transformer_heads=num_heads,
        gripper_width=gripper_width,
        num_images=NUM_IMAGES,
        action_dim=action_dim,
        state_dim=state_dim,
        config=gpt_conf,
    )

    # Load model from checkpoint #
    ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=None)
    params = ckpt["params"]
    batch_stats = ckpt["batch_stats"]
    # Free checkpoint memory
    del ckpt
    import gc
    gc.collect()
    print("Loaded model from checkpoint")

    # Eval #
    model_dict = {
        "model_def": model_def,
        "params": params,
        "batch_stats": batch_stats,
    }
    libero_dir = "/home/skowshik/vla/codebase/envs/LIBERO"
    task_name = "kitchen_scene6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
    libero_cfg = {
        "libero_img_size": image_primary_size,
        "libero_eval_max_steps": 400,
        "gripper_width": gripper_width,
        "history_len": history_length,
        "action_pred_steps": action_pred_steps,
        "use_gt_action": False,
    }
    results, rollout_rbgs = eval_libero10(model_dict, libero_dir, task_name=task_name, num_eval_episodes=1, libero_cfg=libero_cfg)
    save_rgbs_to_gif(rollout_rbgs[0], "model_traj_rgb.gif")

if __name__ == "__main__":
    main()
