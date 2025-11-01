# main.py
from re import L
import tensorflow as tf
# CRITICAL: Disable GPU for TensorFlow to prevent GPU memory allocation during dataset creation
# TensorFlow will allocate GPU memory by default if it detects a GPU, even for dataset operations
tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
tf.random.set_seed(0)

import jax.numpy as jnp
import jax
import torch

import rlds
import clip # To tokenize

import optax
import flax
import wandb
import tqdm
import functools

from data.libero_rlds import LiberoRlds, LiberoRldsConfig, episode_to_windows_with_prefix
from models.bc_simple import generate_attention_mask, BCSimple, GPTConfig

def make_dataset(root_dir: str, info_path: str, img_primary: int, img_wrist: int):
    builder = LiberoRlds(
        config=LiberoRldsConfig(
            name="local_libero_runtime",
            description="Local LIBERO (runtime)",
            root_dir=root_dir,
            info_path=info_path,
            image_primary_size=img_primary,
            image_wrist_size=img_wrist,
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

def process_batch(batch):
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

    # 2) states: concat first 6 dims with last dim (gripper)
    # torch: torch.cat([states_orig[..., :6], states_orig[..., [-1]]], dim=-1)
    states0 = jnp.concatenate(
        [states_orig[..., :6], states_orig[..., -1:]], axis=-1
    )  # shape (..., 7)  ← jnp.concatenate is the JAX/NumPy op :contentReference[oaicite:1]{index=1}

    # 3) binarize the gripper part: (x + 1) // 2
    # torch would do an in-place write; in JAX we rebuild it
    grip = (states0[..., 6:] + 1.0) // 2.0
    states0 = states0.at[..., 6:].set(grip)   # pure, JAXy update :contentReference[oaicite:2]{index=2}

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
    language0 = jnp.asarray(language_tensor.cpu().numpy())

    return images0, states0, actions0, language0

def make_train_step(model_apply, tx): # Need this wrapper as jax.jit expects this format, cannot have model_apply in input to function to be jitted
    @functools.partial(jax.jit)
    def train_step(
                    rng, 
                    params, batch_stats, opt_state, # States to update #
                    # Input Data #
                    images, states, actions, language, attention_mask, batch_targets
                ):
        # Split Key #
        rng, dropout_rng = jax.random.split(rng)

        def loss_fn(params): # Only compute gradients w.r.t. params, not batch_stats
            variables = {"params": params, "batch_stats": batch_stats}
            (action_pred_arm, action_pred_gripper), mutable = model_apply(
                variables,
                images, states, actions, language, attention_mask,
                train=True,
                mutable=['batch_stats'],
                rngs={'dropout': dropout_rng},
            )
            loss_arm = optax.huber_loss(action_pred_arm, batch_targets[:, :, :, :-1]).mean()
            loss_grip = optax.huber_loss(action_pred_gripper, batch_targets[:, :, :, -1:]).mean()
            loss = loss_arm + 0.1 * loss_grip
            return loss, (mutable['batch_stats'], loss_arm, loss_grip)

        (loss, (new_batch_stats, loss_arm, loss_grip)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        batch_stats = new_batch_stats
        
        # Update parameters and state #
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # Compute some statistics #
        grad_norm   = optax.global_norm(grads)
        update_norm = optax.global_norm(updates)
        param_norm  = optax.global_norm(params)

        info_dict = {
            'loss_arm': loss_arm,
            'loss_grip': loss_grip,
            'loss': loss,
            'grad_norm': grad_norm,
            'update_norm': update_norm,
            'param_norm': param_norm,
        }
        
        return params, batch_stats, opt_state, rng, info_dict # RNG is also part of state!
    
    return train_step

def main():
    # CONSTANTS #
    seed = 0
    # Dataloader #
    root_dir = "/data/user_data/skowshik/datasets/libero_pro/libero_10_converted_kitchen_scene6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it/"
    info_path = "./data_info/libero_10_converted_kitchen_scene6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.json"
    image_primary_size = 224
    image_wrist_size = 224
    window_size = 13 # Actual history length is window_size - action_pred_steps
    batch_size = 2
    NUM_IMAGES = 2 # Wrist + Static Camera
    action_pred_steps = 3
    history_length = window_size - action_pred_steps
    action_dim = 7
    state_dim = 7
    # Model #
    hidden_dim = 512
    num_layers = 12
    num_heads = 8
    # Training #
    num_epochs = 10
    learning_rate = 1e-4
    # Toggles #
    USE_WANDB = False
    

    if USE_WANDB:
        # Wandb init #
        run = wandb.init(
            project="vlaft",
            name="bc_run_clean",
            config={
                "lr": learning_rate,
                "batch_size": batch_size,
                "seed": 0,
                "dataset": "LIBERO",
                "model": "BCSimple",
            },
        )

    # DATASET CREATION #
    ds = make_dataset(root_dir, info_path, image_primary_size, image_wrist_size) # Dataset of episodes

    # Create windows and batch
    win_ds = ds.flat_map(lambda ep: episode_to_windows_with_prefix(ep, window_size))
    train_ds = (
        win_ds
        .shuffle(2048)                   # mix windows from different episodes
        .batch(batch_size, drop_remainder=True)
        .prefetch(2)  # Reduced prefetch to limit memory usage (was AUTOTUNE which could be very large)
    )

    # Generate attention mask #
    attention_mask = generate_attention_mask(history_length, NUM_IMAGES + 1 + 1, action_pred_steps) # 2 images + 1 language + 1 state
    attention_mask = jnp.array(attention_mask, dtype=bool) # Fixed static attention mask
    ###########################################################

    # MODEL CREATION #
    gpt_conf = GPTConfig(
        block_size=(history_length + action_pred_steps) * (NUM_IMAGES + 1 + 1 + 3),
        num_layers=num_layers,
        num_heads=num_heads,
        num_embeds=hidden_dim,
        use_bias=True,
        dtype=None,
    )

    model_def = BCSimple(
        sequence_length=history_length,
        input_image_size=image_primary_size,
        action_pred_steps=action_pred_steps,
        transformer_layers=num_layers,
        hidden_dim=hidden_dim,
        transformer_heads=num_heads,
        gripper_width=False,
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
    images0, states0, actions0, language0 = process_batch(init_batch)
    del init_batch  # Clear TF batch from memory
    
    # Condition only to history length
    images0 = images0[:, :, :history_length, ...]
    states0 = states0[:, :history_length, ...]
    language0 = language0[:, :history_length, ...]
    # Pass along to init model
    variables = model_def.init(
        {'params': params_key, 'dropout': dropout_key},
        images0, states0, actions0, language0,
        attention_mask,
        train=False
    )
    params = variables['params']
    batch_stats = variables.get('batch_stats', None)
    
    # Clear init data from GPU memory
    del images0, states0, actions0, language0
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"\n\n\n\n\nParameter count: {param_count / 1e6} M\n\n\n\n\n")

    # Use schedule in Adam optimizer
    tx = optax.adam(learning_rate)
    opt_state = tx.init(params)

    # Create train step function #
    print("Compiling train_step (this may take a moment and use memory)...")
    train_step = make_train_step(model_def.apply, tx)
    
    # Warm up JIT compilation with a dummy step to avoid memory spike during first real step
    print("Warming up JIT compilation...")
    dummy_images = jnp.zeros((batch_size, NUM_IMAGES, history_length, image_primary_size, image_primary_size, 3))
    dummy_states = jnp.zeros((batch_size, history_length, state_dim))
    dummy_actions = jnp.zeros((batch_size, history_length, action_dim))
    dummy_language = jnp.zeros((batch_size, history_length, 77), dtype=jnp.int32)
    dummy_targets = jnp.zeros((batch_size, history_length, action_pred_steps, action_dim))
    _, _, _, _, _ = train_step(
        rng, params, batch_stats, opt_state,
        dummy_images, dummy_states, dummy_actions, dummy_language, attention_mask, dummy_targets
    )
    del dummy_images, dummy_states, dummy_actions, dummy_language, dummy_targets
    
    print("JIT compilation complete. Starting training...")

    train_steps = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for (i, tf_batch) in enumerate(train_ds):
            rng = jax.random.fold_in(rng, i)

            # Process a batch for training
            images0, states0, actions0, language0 = process_batch(tf_batch)
            # Condition only to history length
            images0 = images0[:, :, :history_length, ...]
            states0 = states0[:, :history_length, ...]
            language0 = language0[:, :history_length, ...]
            
            # Generate batch targets
            batch_targets = jnp.concatenate(
                [jnp.expand_dims(actions0[:, j:-action_pred_steps + j, :], axis=-2) for j in range(action_pred_steps)],
                axis=-2
            )
            actions0 = actions0[:, :history_length, ...]
            
            # Train step #
            params, batch_stats, opt_state, rng, info_dict = train_step(
                rng, params, batch_stats, opt_state,
                images0, states0, actions0, language0, attention_mask, batch_targets
            )

            # Log statistics #
            train_steps += 1
            if jax.process_index() == 0:
                if USE_WANDB:
                    wandb.log({
                        'training/loss_arm': float(info_dict['loss_arm']),
                        'training/loss_grip': float(info_dict['loss_grip']),
                        'training/loss': float(info_dict['loss']),
                        'training/grad_norm': float(info_dict['grad_norm']),
                        'training/update_norm': float(info_dict['update_norm']),
                        'training/lr': learning_rate,
                        'training/param_norm': float(info_dict['param_norm']),
                    }, step=int(train_steps))
                else:
                    print(f"Loss Arm: {info_dict['loss_arm']}, Loss Grip: {info_dict['loss_grip']}, Loss: {info_dict['loss']}, Grad Norm: {info_dict['grad_norm']}, Update Norm: {info_dict['update_norm']}, Param Norm: {info_dict['param_norm']}")

if __name__ == "__main__":
    main()
