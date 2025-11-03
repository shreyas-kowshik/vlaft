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
from flax import traverse_util
from flax.core import freeze
from flax.training.common_utils import shard, shard_prng_key

from data.libero_rlds import LiberoRlds, LiberoRldsConfig, episode_to_windows_with_prefix
from models.bc_simple import generate_attention_mask, BCSimple, GPTConfig
from eval_libero_clean import eval_libero10, save_rgbs_to_gif

def make_dataset(root_dir: str, info_path: str, img_primary: int, img_wrist: int, gripper_width: bool = False):
    builder = LiberoRlds(
        config=LiberoRldsConfig(
            name="local_libero_runtime",
            description="Local LIBERO (runtime)",
            root_dir=root_dir,
            info_path=info_path,
            image_primary_size=img_primary,
            image_wrist_size=img_wrist,
            gripper_width=gripper_width,
        )
    )
    # builds TFRecords into ~/tensorflow_datasets/libero_rlds/...
    builder.download_and_prepare()
    ds = builder.as_dataset(split="train")  # dataset of EPISODES
    return ds

def process_language(batch, clip_tokenize=clip.tokenize):
    """
    batch["observation"]["language"]: tf.Tensor of shape (B, T) or (B,) with dtype string/bytes
      e.g. [b"put the yellow mug...", b"open the drawer", ...]
    Returns:
        tokens_bt: torch.LongTensor of shape (B, T, 77)  # 77 is CLIP's context length :contentReference[oaicite:2]{index=2}
        uniq_texts: list[str] (optional, useful for logging)
    """
    lang_tf = batch["observation"]["language"]          # (B, T) tf.string
    lang_np = lang_tf.numpy()                           # -> np.ndarray of bytes

    # normalize to 2D (B, T)
    if lang_np.ndim == 1:
        lang_np = lang_np[:, None]

    B, T = lang_np.shape

    # 1) flatten and decode
    flat_bytes = lang_np.reshape(-1)                    # (B*T,)
    flat_strs = [b.decode("utf-8") for b in flat_bytes]

    # 2) dedupe while keeping order
    str2idx = {}
    uniq_strs = []
    for s in flat_strs:
        if s not in str2idx:
            str2idx[s] = len(uniq_strs)
            uniq_strs.append(s)

    # 3) tokenize once; clip.tokenize already pads/truncates to the CLIP length (77) and returns torch.LongTensor
    #    see openai/CLIP usage: text = clip.tokenize(["a diagram", "a dog"])  :contentReference[oaicite:3]{index=3}
    # Note default tokenizer usees context_length of 77 so will truncate input to 77 tokens
    # 0 is used for padding tokens
    # Assuming input text instruction is less than 77 tokens, batching and padding is automatically handled here
    # Ensure tokenization happens on CPU to avoid GPU memory issues
    with torch.no_grad():
        uniq_tokens = clip_tokenize(uniq_strs, truncate=True)   # (U, 77)
        # Move to CPU explicitly to avoid GPU memory accumulation
        uniq_tokens = uniq_tokens.cpu()

    # 4) map back to (B*T, 77)
    idxs = torch.tensor([str2idx[s] for s in flat_strs], dtype=torch.long, device='cpu')
    tokens_bt = uniq_tokens[idxs]                           # (B*T, 77)

    # 5) reshape to (B, T, 77)
    tokens_bt = tokens_bt.view(B, T, -1)

    return tokens_bt, uniq_strs

def process_batch(batch, gripper_width: bool = False):
    """
    batch: dict coming from your tf.data pipeline (already windowed + batched)
           shapes (~): 
             batch["observation"]["rgb_static"]   -> (B, T, H, W, 3)
             batch["observation"]["rgb_gripper"]  -> (B, T, H, W, 3)
             batch["observation"]["robot_obs"]    -> (B, T, 15)
             batch["action"]                      -> (B, T, A)
    returns:
        images0  -> (B, 2, T, H, W, 3)   # static + wrist
        states0  -> (B, T, 7)            # 6 pose + 1 gripper (0/1)
        actions0 -> (B, T, A)            # last dims in {0,1}
    """
    # 1) tf.Tensor -> numpy -> jax.Array (convert to numpy first to avoid GPU allocation during conversion)
    # Normalize images from [0, 255] uint8 to [0, 1] float32 for ResNet encoder
    rgb_static  = jnp.asarray(batch["observation"]["rgb_static"].numpy(), dtype=jnp.float32) / 255.0
    wrist_rgb   = jnp.asarray(batch["observation"]["rgb_gripper"].numpy(), dtype=jnp.float32) / 255.0
    states_orig = jnp.asarray(batch["observation"]["robot_obs"].numpy(), dtype=jnp.float32)
    actions0    = jnp.asarray(batch["action"].numpy(), dtype=jnp.float32)
    step_ids0   = jnp.asarray(batch["step_id"].numpy(), dtype=jnp.int32)

    # 2) states: concat first 6 dims with last dim (gripper)
    # torch: torch.cat([states_orig[..., :6], states_orig[..., [-1]]], dim=-1)
    if not gripper_width:
        states0 = jnp.concatenate(
            [states_orig[..., :6], states_orig[..., -1:]], axis=-1
        )  # shape (..., 7)  ← jnp.concatenate is the JAX/NumPy op :contentReference[oaicite:1]{index=1}

        # 3) binarize the gripper part: (x + 1) // 2
        # torch would do an in-place write; in JAX we rebuild it
        grip = (states0[..., 6:] + 1.0) // 2.0
        states0 = states0.at[..., 6:].set(grip)   # pure, JAXy update :contentReference[oaicite:2]{index=2}
    else:
        states0 = jnp.concatenate(
            [states_orig[..., :6], states_orig[..., -2:]], axis=-1
        ) # (..., 8)

    # 4) stack cameras
    # torch: torch.cat([rgb_static.unsqueeze(1), wrist_rgb.unsqueeze(1)], dim=1)
    rgb_static_1 = jnp.expand_dims(rgb_static, axis=1)   # (B,1,T,H,W,3)  ← unsqueeze in JAX :contentReference[oaicite:3]{index=3}
    wrist_rgb_1  = jnp.expand_dims(wrist_rgb,  axis=1)   # (B,1,T,H,W,3)
    images0 = jnp.concatenate([rgb_static_1, wrist_rgb_1], axis=1)  # (B,2,T,H,W,3)

    # 5) binarize action tail the same way as in torch
    if actions0.shape[-1] > 6:
        act_tail = (actions0[..., 6:] + 1.0) // 2.0
        actions0 = actions0.at[..., 6:].set(act_tail)
    
    language_tensor, _ = process_language(batch)
    # Convert torch language_tensor to numpy first, then jax array (avoids GPU memory from torch)
    language0 = jnp.asarray(language_tensor.cpu().numpy(), dtype=jnp.int32)

    return images0, states0, actions0, language0, step_ids0

def make_train_step(model_apply, tx): # Need this wrapper as jax.jit expects this format, cannot have model_apply in input to function to be jitted
    # PMAP Diff #
    @functools.partial(jax.pmap, axis_name='data')
    @functools.partial(jax.jit)
    def train_step(
                    rng, 
                    params, batch_stats, opt_state, # States to update #
                    # Input Data #
                    images, states, actions, language, attention_mask, batch_targets, step_ids
                ):
        # print("Train step print for JIT check: {}".format(rng)) # If JIT is enabled, this should not be printed as JIT should ignore this line
        # Split Key #
        rng, dropout_rng = jax.random.split(rng)

        def loss_fn(params): # Only compute gradients w.r.t. params, not batch_stats
            variables = {"params": params, "batch_stats": batch_stats}
            (action_pred_arm, action_pred_gripper), mutable = model_apply(
                variables,
                images, states, language, attention_mask, step_ids,
                train=True,
                mutable=['batch_stats'],
                rngs={'dropout': dropout_rng},
            )
            loss_arm = optax.huber_loss(action_pred_arm, batch_targets[:, :, :, :-1]).mean()
            # loss_grip = optax.huber_loss(action_pred_gripper, batch_targets[:, :, :, -1:]).mean()
            loss_grip = optax.sigmoid_binary_cross_entropy(action_pred_gripper, batch_targets[:, :, :, -1:]).mean()
            loss = loss_arm + 0.2 * loss_grip

            # Get per dimension L1 loss for arm and gripper and return in info_dict #
            l1_loss_arm0 = jnp.abs(action_pred_arm[:, :, :, 0] - batch_targets[:, :, :, 0]).mean()
            l1_loss_arm1 = jnp.abs(action_pred_arm[:, :, :, 1] - batch_targets[:, :, :, 1]).mean()
            l1_loss_arm2 = jnp.abs(action_pred_arm[:, :, :, 2] - batch_targets[:, :, :, 2]).mean()
            l1_loss_arm3 = jnp.abs(action_pred_arm[:, :, :, 3] - batch_targets[:, :, :, 3]).mean()
            l1_loss_arm4 = jnp.abs(action_pred_arm[:, :, :, 4] - batch_targets[:, :, :, 4]).mean()
            l1_loss_arm5 = jnp.abs(action_pred_arm[:, :, :, 5] - batch_targets[:, :, :, 5]).mean()
            l1_loss_grip = jnp.abs(jax.nn.sigmoid(action_pred_gripper[:, :, :, 6]) - batch_targets[:, :, :, 6]).mean()
            
            return loss, (mutable['batch_stats'], 
                            loss_arm, loss_grip, l1_loss_arm0, l1_loss_arm1, l1_loss_arm2, l1_loss_arm3, l1_loss_arm4, l1_loss_arm5, l1_loss_grip)

        (loss, vals), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # PMAP Diff #
        grads = jax.lax.pmean(grads, axis_name='data')
        loss = jax.lax.pmean(loss, axis_name='data')
        new_batch_stats = vals[0]
        new_batch_stats = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, 'data'), new_batch_stats)
        
        batch_stats = new_batch_stats
        loss_arm = jax.lax.pmean(vals[1], axis_name='data')
        loss_grip = jax.lax.pmean(vals[2], axis_name='data')
        l1_loss_arm0 = jax.lax.pmean(vals[3], axis_name='data')
        l1_loss_arm1 = jax.lax.pmean(vals[4], axis_name='data')
        l1_loss_arm2 = jax.lax.pmean(vals[5], axis_name='data')
        l1_loss_arm3 = jax.lax.pmean(vals[6], axis_name='data')
        l1_loss_arm4 = jax.lax.pmean(vals[7], axis_name='data')
        l1_loss_arm5 = jax.lax.pmean(vals[8], axis_name='data')
        l1_loss_grip = jax.lax.pmean(vals[9], axis_name='data')
        
        # Update parameters and state #
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # Compute some statistics #
        grad_norm   = jax.lax.pmean(optax.global_norm(grads), axis_name='data')
        update_norm = jax.lax.pmean(optax.global_norm(updates), axis_name='data')
        param_norm  = jax.lax.pmean(optax.global_norm(params), axis_name='data')
        vision_param_norm = jax.lax.pmean(optax.global_norm(params['image_encoder']), axis_name='data')

        info_dict = {
            'loss_arm': loss_arm,
            'loss_grip': loss_grip,
            'loss': loss,
            'l1_loss_arm0': l1_loss_arm0,
            'l1_loss_arm1': l1_loss_arm1,
            'l1_loss_arm2': l1_loss_arm2,
            'l1_loss_arm3': l1_loss_arm3,
            'l1_loss_arm4': l1_loss_arm4,
            'l1_loss_arm5': l1_loss_arm5,
            'l1_loss_grip': l1_loss_grip,
            'grad_norm': grad_norm,
            'update_norm': update_norm,
            'param_norm': param_norm,
            'vision_param_norm': vision_param_norm,
        }
        
        return params, batch_stats, opt_state, rng, info_dict # RNG is also part of state!
    
    return train_step

def make_tx(params):
    def label(path, _):
        root = path[0]
        if root in ("clip", "resnet"):
            return "frozen"
        return "trainable"

    param_labels = freeze(traverse_util.path_aware_map(label, params))
    tx = optax.multi_transform(
        {
            "trainable": optax.adam(1e-4),
            "frozen": optax.set_to_zero(),
        },
        param_labels,
    )
    return tx

def main():
    # CONSTANTS #
    seed = 0
    # Dataloader #
    root_dir = "/data/user_data/skowshik/datasets/libero_pro/libero_10_converted_kitchen_scene6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it/"
    info_path = "./data_info/libero_10_converted_kitchen_scene6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.json"
    CKPT_DIR = "/data/user_data/skowshik/checkpoints/libero_10_converted_kitchen_scene6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it/one_demo_ckpt_dir"
    if os.path.exists(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)
    os.makedirs(CKPT_DIR)
    image_primary_size = 100
    image_wrist_size = 100
    window_size = 8 # Actual history length is window_size - action_pred_steps
    batch_size = 128 * int(jax.device_count()) # Batch size per device
    NUM_IMAGES = 2 # Wrist + Static Camera
    action_pred_steps = 3
    history_length = window_size - action_pred_steps
    action_dim = 7
    state_dim = 7
    gripper_width = True
    if gripper_width:
        state_dim = 8
    train_ds_len = 100 * int(jax.device_count()) # max(1, 20 // batch_size)
    # Model #
    hidden_dim = 512
    num_layers = 8
    num_heads = 8
    dropout_rate = 0.2
    # Training #
    num_epochs = 200
    learning_rate = 1e-3
    use_lr_schedule = True
    num_eval_episodes = 10
    # Toggles #
    USE_WANDB = True
    EVAL_AFTER_EPOCH = True
    DEBUG_DATA_SUBSET = False
    

    if USE_WANDB:
        # Wandb init #
        run = wandb.init(
            project="vlaft",
            name="bc_run_debug_eval_with_train_multi_gpu",
            config={
                "lr": learning_rate,
                "batch_size": batch_size,
                "seed": 0,
                "dataset": "LIBERO",
                "model": "BCSimple",
            },
        )
    
    # PMAP Diff #
    # Print device statistics #
    print("jax.local_devices():", jax.local_devices())
    print("jax.device_count():", jax.device_count())

    # DATASET CREATION #
    ds = make_dataset(root_dir, info_path, image_primary_size, image_wrist_size, gripper_width=gripper_width) # Dataset of episodes

    # Create windows and batch
    # CHECKPOINT: Dataloading and windowing is working correctly #
    win_ds = ds.flat_map(lambda ep: episode_to_windows_with_prefix(ep, window_size))
    train_ds = (
        win_ds
        # .shuffle(2048)                   # mix windows from different episodes
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)  # Reduced prefetch to limit memory usage (was AUTOTUNE which could be very large)
    )
    if DEBUG_DATA_SUBSET:
        train_ds = train_ds.take(20)
    debug_batch = next(iter(train_ds))
    images0, states0, actions0, language0, step_ids0 = process_batch(debug_batch)
    # CHECKPOINT: Breakpoint above to check if batching and dataloading is working correctly #

    # Generate attention mask #
    # CHECKPOINT: Attention mask generation is working correctly #
    attention_mask = generate_attention_mask(history_length, NUM_IMAGES + 1 + 1, action_pred_steps) # 2 images + 1 language + 1 state
    attention_mask = jnp.array(attention_mask, dtype=bool) # Fixed static attention mask
    # breakpoint()
    # CHECKPOINT: Breakpoint above to check if attention mask generation is working correctly #
    ###########################################################

    # MODEL CREATION #
    gpt_conf = GPTConfig(
        block_size=(history_length + action_pred_steps) * (NUM_IMAGES + 1 + 1 + action_pred_steps),
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
    # breakpoint()

    # Init model (JIT-only)
    rng = jax.random.PRNGKey(seed)
    rng, params_key, dropout_key = jax.random.split(rng, 3)
    # Get some initial data to pass to the model
    print("Initializing model...")
    init_batch = next(iter(train_ds.take(1)))
    images0, states0, actions0, language0, step_ids0 = process_batch(init_batch, gripper_width=gripper_width)   
    del init_batch  # Clear TF batch from memory
    
    # Condition only to history length
    images0 = images0[:, :, :history_length, ...]
    states0 = states0[:, :history_length, ...]
    language0 = language0[:, :history_length, ...]
    step_ids0 = step_ids0[:, :history_length, ...]
    # Pass along to init model
    variables = model_def.init(
        {'params': params_key, 'dropout': dropout_key},
        images0, states0, language0,
        attention_mask,
        step_ids0,
        train=False
    )
    params = variables['params']
    batch_stats = variables.get('batch_stats', None)
    # breakpoint()
    
    # Clear init data from GPU memory
    del images0, states0, actions0, language0, step_ids0
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"\n\n\n\n\nParameter count: {param_count / 1e6} M\n\n\n\n\n")

    # Use schedule in Adam optimizer
    if use_lr_schedule:
        total_steps = num_epochs * train_ds_len
        warmup_steps = 30 # max(1, int(0.01 * total_steps))  # 1% warmup (adjust to match Seer config)
        decay_steps  = max(1, total_steps - warmup_steps)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,                 # start at 0
            peak_value=float(learning_rate),
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=1e-5
        )
        # params['image_encoder'] = freeze(params['image_encoder'])
        tx = optax.adam(lr_schedule)
        # tx = make_tx(params)
    else:
        tx = optax.adam(learning_rate)
    
    # PMAP Diff #
    # opt_state = tx.init(params) # Move this below
    # breakpoint()

    # PMAP Diff #
    # Split stuff across devices #
    # model_def = flax.jax_utils.replicate(model_def, devices=jax.local_devices()) # No need to replicate this
    # tx = flax.jax_utils.replicate(tx, devices=jax.local_devices()) # No need to replicate this
    opt_state = flax.jax_utils.replicate(tx.init(flax.jax_utils.unreplicate(params)), devices=jax.local_devices())
    params = flax.jax_utils.replicate(params, devices=jax.local_devices())
    batch_stats = flax.jax_utils.replicate(batch_stats, devices=jax.local_devices())
    attention_mask = flax.jax_utils.replicate(attention_mask, devices=jax.local_devices())

    # Create train step function #
    print("Compiling train_step (this may take a moment and use memory)...")
    train_step = make_train_step(model_def.apply, tx)

    train_steps = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for (i, tf_batch) in enumerate(train_ds):
            # PMAP Diff #
            # rng = jax.random.fold_in(rng, train_steps + 1)
            step_rng = jax.random.fold_in(rng, train_steps + 1)          # per-step
            step_rng = jax.random.fold_in(step_rng, jax.process_index())  # per-host (multi-host safety)

            # Process a batch for training
            images0, states0, actions0, language0, step_ids0 = process_batch(tf_batch, gripper_width=gripper_width)
            # Condition only to history length
            images0 = images0[:, :, :history_length, ...]
            states0 = states0[:, :history_length, ...]
            language0 = language0[:, :history_length, ...]
            step_ids0 = step_ids0[:, :history_length, ...]

            # Generate batch targets
            batch_targets = jnp.concatenate(
                [jnp.expand_dims(actions0[:, j:-action_pred_steps + j, :], axis=-2) for j in range(action_pred_steps)],
                axis=-2
            )
            actions0 = actions0[:, :history_length, ...]
            
            # Train step #
            # PMAP Diff #
            # Use new_rng here, can skip using it in future
            params, batch_stats, opt_state, new_rng, info_dict = train_step(
                # rng,
                # PMAP Diff #
                shard_prng_key(step_rng), # Use step_rng here derived from base rng value
                params, batch_stats, opt_state,
                # PMAP Diff #
                shard(images0),
                shard(states0),
                shard(actions0),
                shard(language0),
                attention_mask,
                shard(batch_targets),
                shard(step_ids0)
            )

            # Log statistics #
            train_steps += 1
            if jax.process_index() == 0:
                if USE_WANDB:
                    lr_log = learning_rate
                    if use_lr_schedule:
                        lr_log = float(jax.device_get(lr_schedule(int(train_steps))))
                    wandb.log({
                        'training/loss_arm': float(info_dict['loss_arm'][0]),
                        'training/loss_grip': float(info_dict['loss_grip'][0]),
                        'training/loss': float(info_dict['loss'][0]),
                        'training/l1_loss_arm0': float(info_dict['l1_loss_arm0'][0]),
                        'training/l1_loss_arm1': float(info_dict['l1_loss_arm1'][0]),
                        'training/l1_loss_arm2': float(info_dict['l1_loss_arm2'][0]),
                        'training/l1_loss_arm3': float(info_dict['l1_loss_arm3'][0]),
                        'training/l1_loss_arm4': float(info_dict['l1_loss_arm4'][0]),
                        'training/l1_loss_arm5': float(info_dict['l1_loss_arm5'][0]),
                        'training/l1_loss_grip': float(info_dict['l1_loss_grip'][0]),
                        'training/grad_norm': float(info_dict['grad_norm'][0]),
                        'training/update_norm': float(info_dict['update_norm'][0]),
                        'training/vision_param_norm': float(info_dict['vision_param_norm'][0]),
                        'training/lr': lr_log,
                        'training/param_norm': float(info_dict['param_norm'][0]),
                        'training/epoch': epoch,
                        'training/train_steps': train_steps,
                    }, step=int(train_steps))
                else:
                    print(f"Loss Arm: {info_dict['loss_arm'][0]}, Loss Grip: {info_dict['loss_grip'][0]}, Loss: {info_dict['loss'][0]}, Grad Norm: {info_dict['grad_norm'][0]}, Update Norm: {info_dict['update_norm'][0]}, Param Norm: {info_dict['param_norm'][0]}")

        # Eval #
        if EVAL_AFTER_EPOCH and jax.process_index() == 0: # Only run eval on main process
            # PMAP Diff #
            params_host = jax.device_get(flax.jax_utils.unreplicate(params))
            batch_stats_host = jax.device_get(flax.jax_utils.unreplicate(batch_stats))
            model_dict = {
                "model_def": model_def,
                "params": params_host,
                "batch_stats": batch_stats_host,
            }
            libero_dir = "/home/skowshik/vla/codebase/envs/LIBERO"
            task_name = "kitchen_scene6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
            libero_cfg = {
                "libero_img_size": image_primary_size,
                "libero_eval_max_steps": 400,
                "gripper_width": gripper_width,
                "history_length": history_length,
                "action_pred_steps": action_pred_steps,
            }
            results, rollout_rbgs = eval_libero10(model_dict, libero_dir, task_name=task_name, num_eval_episodes=num_eval_episodes, libero_cfg=libero_cfg)
            # Sort results and rollouts by values of results
            sorted_indices = np.argsort(results)
            results = [results[i] for i in sorted_indices]
            rollout_rbgs = [rollout_rbgs[i] for i in sorted_indices]
            print("Results: {}".format(results))
            # Log best rollout rgb .gif to wandb
            if USE_WANDB:
                # Log result success rate
                success_rate = np.mean(results)
                wandb.log({
                    'evaluation/success_rate': success_rate,
                }, step=train_steps)
                # Log best rollout rgb .gif to wandb
                save_rgbs_to_gif(rollout_rbgs[-1], "best_rollout_rgb.gif")
                wandb.log({
                    "rollout": wandb.Video("best_rollout_rgb.gif", format="gif", fps=11)  # fps is ignored for file paths but OK
                }, step=train_steps)

    if jax.process_index() == 0: # Only save checkpoint on main process
        print("Saving checkpoint to {}...".format(CKPT_DIR))
        # PMAP Diff #
        params_host = jax.device_get(flax.jax_utils.unreplicate(params))
        batch_stats_host = jax.device_get(flax.jax_utils.unreplicate(batch_stats))
        opt_state_host = jax.device_get(flax.jax_utils.unreplicate(opt_state))
        rng_host = jax.device_get(flax.jax_utils.unreplicate(new_rng))
        ckpt_target = {
            "params": params_host,
            "batch_stats": batch_stats_host,
            "opt_state": opt_state_host,
            "rng": rng_host,
        }
        checkpoints.save_checkpoint(
            ckpt_dir=CKPT_DIR,
            target=ckpt_target,
            step=train_steps,        # or epoch + 1, your call
            overwrite=True,          # replace latest
            keep=3,                  # keep last 3
        )
    
        if USE_WANDB:
            run.finish()

if __name__ == "__main__":
    main()