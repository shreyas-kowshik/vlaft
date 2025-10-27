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
from models.gpt2_jax import GPT, GPTConfig

def generate_attention_mask(K, num_A, num_B):
    # num_A: 1+1+self.NUM_RESAMPLER_QUERY*2+1*2
    # num_A: text, state, image_embedding, image_cls_token_embedding
    # num_B: self.NUM_OBS_TOKEN+self.action_pred_steps
    # num_B: obs_tokens(if exists), action_pred_token, state_pred_token (if exists)
    sequence_length = (num_A + num_B) * K
    attention_mask = np.ones((sequence_length, sequence_length))
    for i in range(K):
        start_index = i * (num_A + num_B)
        end_index = start_index + num_A + num_B
        
        # the i-th sub-sequence can not attend to the sub-sequences that after the i-th
        attention_mask[start_index:end_index, end_index:] = 0
        
        # the sub-sub-sequence B can not be attended to
        attention_mask[:, start_index+num_A:end_index] = 0
            
    return attention_mask

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
    config: GPTConfig = None

    def setup(self):
        self.image_encoder = fm.ResNet18(output='activations', pretrained='imagenet', normalize=True)
        self.image_projector = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.normal(0.02))
        self.state_encoder = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.normal(0.02))
        self.clip = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_projector = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.normal(0.02))
        self.timestep_embedding = TimestepEmbedder(self.hidden_dim)
        self.action_embedding = self.param(
            "actino_embedding",
            normal(stddev=0.02),      # init_fn(key, shape, dtype) -> array
            (3, self.hidden_dim)
        )
        self.action_projector_arm = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.normal(0.02)),
            nn.relu,
            nn.Dense(self.action_dim - 1, kernel_init=nn.initializers.normal(0.02)),
            nn.tanh,
        ])
        self.action_projector_gripper = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.normal(0.02)), 
            nn.relu,
            nn.Dense(1, kernel_init=nn.initializers.normal(0.02)), 
            nn.sigmoid
        ])
        self.transformer = GPT(self.config)
        
    @nn.compact
    def __call__(self, images, states, actions, text_tokens, attention_mask, train=False):
        # (x = (B, H, W, C) image, t = (B,) timesteps, y = (B,) class labels)
        # images = (B, num_images, T, H, W, C)
        # states = (B, T, state_dim)
        # actions = (B, T, action_dim)
        # breakpoint()
        B, num_images, T, C, H, W = images.shape
        images = images.reshape(-1, H, W, C)
        image_emb = self.image_encoder(images)['block4_1']
        image_emb = image_emb.reshape(B, T, num_images, -1)
        image_emb = self.image_projector(image_emb) # (B, T, num_images, hidden_dim)

        state_emb = self.state_encoder(states) # (B, T, hidden_dim)

        text_tokens = text_tokens.reshape(B, -1)
        text_emb = self.clip.get_text_features(text_tokens, params=self.clip.params, train=False) # (B * T, hidden_dim)
        # text_emb = text_emb.reshape(B, -1)
        text_emb = self.text_projector(text_emb) # (B, T, hidden_dim)
        text_emb = jnp.repeat(jnp.expand_dims(text_emb, axis=1), T, axis=1) # (B, T, hidden_dim)

        # Add global timestep embedding to images, state, text
        timestep_embedding = self.timestep_embedding(jnp.arange(T)) # (T, hidden_dim)
        timestep_embedding = timestep_embedding.reshape(1, T, -1)
        image_emb = image_emb + jnp.expand_dims(timestep_embedding, axis=2)
        state_emb = state_emb + timestep_embedding
        text_emb = text_emb + timestep_embedding

        action_emb = self.action_embedding.reshape(1, 1, 3, -1)
        action_emb = action_emb + timestep_embedding.reshape(1, T, 1, -1)

        # Concatenate all embeddings and pass through transformer
        transformer_input = jnp.concatenate([image_emb, 
                            jnp.expand_dims(state_emb, axis=2), 
                            jnp.expand_dims(text_emb, axis=2),
                            jnp.repeat(action_emb, B, axis=0)], axis=2) # (B, T, num_images + 1 + 1 + 3, hidden_dim)
        
        # Generate attention mask
        # attention_mask = generate_attention_mask(self.sequence_length, self.num_images + 1 + 1, self.action_pred_steps)
        # breakpoint()
        transformer_input = transformer_input.reshape(B, -1, self.hidden_dim) # (B, T * (num_images + 1 + 1 + 3), hidden_dim)
        attention_mask = jnp.expand_dims(attention_mask, axis=0)
        transformer_output = self.transformer(transformer_input, attention_mask, deterministic=not train)
        # breakpoint()
        transformer_output = transformer_output.reshape(B, T, -1, self.hidden_dim)
        action_pred_tokens = transformer_output[:, :, -self.action_pred_steps:, :]
        action_pred_arm = self.action_projector_arm(action_pred_tokens)
        action_pred_gripper = self.action_projector_gripper(action_pred_tokens)
        # breakpoint()
        return action_pred_arm, action_pred_gripper

if __name__ == "__main__":
    config = GPTConfig()
    model = BCSimple(
        10, # sequence_length : int = 10
        224, # input_image_size : int = 224
        3, # action_pred_steps : int = 3
        12, # transformer_layers: int = 12
        768, # hidden_dim: int = 768
        12, # transformer_heads: int = 12
        False, # gripper_width: bool = False
        2, # num_images: int = 2
        7, # action_dim: int = 7
        7, # state_dim: int = 7
        config
    )

    rng = jax.random.PRNGKey(0)

    rng, params_rng, dropout_rng = jax.random.split(rng, 3)
    example_images = jax.random.normal(rng, (2, 2, 10, 224, 224, 3))
    example_states = jax.random.normal(rng, (2, 10, 7))
    example_actions = jax.random.normal(rng, (2, 10, 7))
    example_text_tokens = jnp.ones((2, 10, 77))

    attention_mask = generate_attention_mask(model.sequence_length, model.num_images + 1 + 1, model.action_pred_steps)
    attention_mask = jnp.array(attention_mask, dtype=bool)
    variables = model.init({'params': params_rng, 'dropout': dropout_rng}, 
                        example_images, example_states, example_actions, example_text_tokens,
                        attention_mask,
                        train=False
                        )
    params = variables['params']

    batch_stats = variables['batch_stats']

    rng, apply_dropout_rng = jax.random.split(rng)
    (action_pred_arm, action_pred_gripper), mutable = model.apply(variables,
                        example_images, example_states, example_actions, example_text_tokens,
                        attention_mask,
                        train=True, 
                        mutable=['batch_stats'],
                        rngs={'dropout': apply_dropout_rng})
    # print(output.shape)
