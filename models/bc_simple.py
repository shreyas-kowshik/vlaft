import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import tree_util
import optax
import functools
from typing import Any, Callable
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import flaxmodels as fm
import math
from flax.linen.initializers import normal

import clip
from transformers import AutoProcessor, FlaxCLIPModel
from gpt2_jax import GPT, GPTConfig

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        return x

    # t is between [0, max_period]. It's the INTEGER timestep, not the fractional (0,1).;
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp( -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding

class BCSimple(nn.Module):
    '''
    Simple BC model with a Transformer backbone.
    Based on the Seer agent architecture with a focus on minimalism

    Given a history of image observations and proprioceptive state, predict the next action chunk
    '''
    sequence_length : int = 10
    input_image_size : int = 224
    action_pred_steps : int = 3
    transformer_layers: int = 12
    hidden_dim: int = 768
    transformer_heads: int = 12
    gripper_width: bool = False
    num_images: int = 2
    action_dim: int = 7
    state_dim: int = 7
    config = None

    def setup(self):
        self.image_encoder = fm.ResNet18(output='logits', pretrained='imagenet')
        self.image_projector = nn.Dense(1000, self.hidden_dim)
        self.state_encoder = nn.Dense(self.state_dim, self.hidden_dim)
        self.clip = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_projector = nn.Dense(512, self.hidden_dim)
        self.timestep_embedding = TimestepEmbedder(self.hidden_dim)
        self.action_embedding = self.param(
            "embedding",
            normal(stddev=0.02),      # init_fn(key, shape, dtype) -> array
            (3, self.hidden_dim)
        )
        self.action_projector = nn.Dense(self.action_dim, self.hidden_dim)
        self.transformer = GPT(self.config)
        
    @nn.compact
    def __call__(self, images, states, actions, text_tokens, train=False):
        # (x = (B, H, W, C) image, t = (B,) timesteps, y = (B,) class labels)
        # images = (B, num_images, T, H, W, C)
        # states = (B, T, state_dim)
        # actions = (B, T, action_dim)
        B, num_images, T, H, W, C = images.shape
        images = images.reshape(-1, H, W, C)
        image_emb = self.image_encoder(images)
        image_emb = image_emb.reshape(B, T, num_images, -1)
        breakpoint()
        image_emb = self.image_projector(image_emb) # (B, T, num_images, hidden_dim)
        breakpoint()

        state_emb = self.state_encoder(states) # (B, T, hidden_dim)

        text_tokens = text_tokens.reshape(-1, 77)
        text_emb = self.clip.get_text_features(text_tokens) # (B * T, hidden_dim)
        text_emb = text_emb.reshape(B, T, -1)
        text_emb = self.text_projector(text_emb) # (B, T, hidden_dim)

        # Add global timestep embedding to images, state, text
        timestep_embedding = self.timestep_embedding(jnp.arange(T)) # (T, hidden_dim)
        timestep_embedding = timestep_embedding.reshape(1, T, -1)
        image_emb = image_emb + timestep_embedding
        state_emb = state_emb + timestep_embedding
        text_emb = text_emb + timestep_embedding

        action_emb = action_emb.reshape(1, 1, 3, -1)
        action_emb = action_emb + timestep_embedding.reshape(1, T, 1, -1)

        # Concatenate all embeddings and pass through transformer
        transformer_input = jnp.concatenate([jnp.expand_dims(image_emb, axis=2), 
                            jnp.expand_dims(state_emb, axis=2), 
                            jnp.expand_dims(text_emb, axis=2),
                            action_emb], axis=2) # (B, T, num_images + 1 + 1 + 3, hidden_dim)
        
        transformer_output = self.transformer(transformer_input, deterministic=not train)
        action_pred = self.action_projector(transformer_output)
        return action_pred

if __name__ == "__main__":
    config = GPTConfig()
    model = BCSimple(config)

    rng = jax.random.PRNGKey(0)

    rng, params_rng, dropout_rng = jax.random.split(rng, 3)
    example_images = jax.random.normal(rng, (2, 2, 10, 224, 224, 3))
    example_states = jax.random.normal(rng, (2, 10, 7))
    example_actions = jax.random.normal(rng, (2, 10, 7))
    example_text_tokens = jnp.ones((2, 10, 77))
    params = model.init({'params': params_rng, 'dropout': dropout_rng}, 
                        example_images, example_states, example_actions, example_text_tokens,
                        train=False
                        )
    rng, apply_dropout_rng = jax.random.split(rng)
    output = model.apply(params,
                        example_images, example_states, example_actions, example_text_tokens,
                        train=True, rngs={'dropout': apply_dropout_rng})
    print(output.shape)
    breakpoint()