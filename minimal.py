###############################
#
#  Minimal JAX Tutorial with CNN and TrainState
#
###############################

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import tree_util
import optax
import functools
from typing import Any, Callable
import numpy as np

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

# TrainState class for managing model parameters and optimizer state
class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Callable = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def: nn.Module, params, tx=None, **kwargs):
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1, apply_fn=model_def.apply, model_def=model_def, params=params,
            tx=tx, opt_state=opt_state, **kwargs,
        )

    # Call model_def.apply_fn
    def __call__(self, *args, params=None, method=None, **kwargs):
        if params is None:
            params = self.params
        variables = {"params": params}
        if isinstance(method, str):
            method = getattr(self.model_def, method)
        return self.apply_fn(variables, *args, method=method, **kwargs)
    
    # Shortcut for above
    def do(self, method):
        return functools.partial(self, method=method)

    def apply_gradients(self, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs)

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

# Simple CNN model
class SimpleCNN(nn.Module):
    num_classes: int = 10
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # First conv block
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        breakpoint()
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Second conv block
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Third conv block
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        
        # Dense layers
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not train)(x)
        x = nn.Dense(features=self.num_classes)(x)
        
        return x

# Training function
def train_step(state, batch_images, batch_labels):
    """Single training step using TrainState."""
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch_images, train=True)
        loss = optax.softmax_cross_entropy(logits, batch_labels).mean()
        accuracy = (jnp.argmax(logits, axis=-1) == jnp.argmax(batch_labels, axis=-1)).mean()
        return loss, {'accuracy': accuracy}
    
    # Apply gradients using TrainState
    new_state, info = state.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
    return new_state, info

# Evaluation function
def eval_step(state, batch_images, batch_labels):
    """Evaluation step."""
    logits = state(batch_images, train=False)
    loss = optax.softmax_cross_entropy(logits, batch_labels).mean()
    accuracy = (jnp.argmax(logits, axis=-1) == jnp.argmax(batch_labels, axis=-1)).mean()
    return {'loss': loss, 'accuracy': accuracy}

def main():
    """Main training loop with dummy data."""
    print("JAX Tutorial: Simple CNN with TrainState")
    print("=" * 50)
    
    # Set random seed
    rng = jax.random.PRNGKey(42)
    np.random.seed(42)
    
    # Create dummy data (CIFAR-10 style)
    batch_size = 32
    image_size = 32
    num_channels = 3
    num_classes = 10
    
    # Generate random dummy data
    rng, data_rng = jax.random.split(rng)
    dummy_images = jax.random.normal(data_rng, (batch_size, image_size, image_size, num_channels))
    dummy_labels = jax.random.randint(data_rng, (batch_size,), 0, num_classes)
    dummy_labels_onehot = jax.nn.one_hot(dummy_labels, num_classes)
    
    print(f"Data shapes:")
    print(f"  Images: {dummy_images.shape}")
    print(f"  Labels: {dummy_labels.shape}")
    print(f"  Labels onehot: {dummy_labels_onehot.shape}")
    
    # Create model
    model = SimpleCNN(num_classes=num_classes)
    
    # Initialize parameters
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, dummy_images, train=True)['params']
    
    # Create optimizer
    tx = optax.adam(learning_rate=0.001)
    
    # Create TrainState
    state = TrainState.create(model, params, tx=tx)
    
    print(f"\nModel initialized with {sum(x.size for x in jax.tree_util.tree_leaves(params))} parameters")
    print(f"Initial step: {state.step}")
    
    # Training loop
    num_steps = 100
    print(f"\nStarting training for {num_steps} steps...")
    
    for step in range(num_steps):
        # Generate new random batch for each step (simulating data loading)
        rng, data_rng = jax.random.split(rng)
        batch_images = jax.random.normal(data_rng, (batch_size, image_size, image_size, num_channels))
        batch_labels = jax.random.randint(data_rng, (batch_size,), 0, num_classes)
        batch_labels_onehot = jax.nn.one_hot(batch_labels, num_classes)
        
        # Training step
        state, train_info = train_step(state, batch_images, batch_labels_onehot)
        
        # Log progress
        if step % 20 == 0 or step == num_steps - 1:
            # Evaluation on same batch
            eval_info = eval_step(state, batch_images, batch_labels_onehot)
            
            print(f"Step {step:3d}: "
                  f"Train Acc: {train_info['accuracy']:.3f}, "
                  f"Eval Loss: {eval_info['loss']:.3f}, "
                  f"Eval Acc: {eval_info['accuracy']:.3f}")
    
    print(f"\nTraining completed!")
    print(f"Final step: {state.step}")
    
    # Demonstrate model inference
    print(f"\nDemonstrating inference...")
    rng, data_rng = jax.random.split(rng)
    test_images = jax.random.normal(data_rng, (4, image_size, image_size, num_channels))
    test_labels = jax.random.randint(data_rng, (4,), 0, num_classes)
    
    # Get predictions
    logits = state(test_images, train=False)
    predictions = jnp.argmax(logits, axis=-1)
    
    print(f"Test images shape: {test_images.shape}")
    print(f"True labels: {test_labels}")
    print(f"Predictions: {predictions}")
    print(f"Accuracy: {(predictions == test_labels).mean():.3f}")
    
    # Demonstrate state saving/loading
    print(f"\nDemonstrating state management...")
    saved_data = {
        'params': state.params,
        'opt_state': state.opt_state,
        'step': state.step,
    }
    
    # Create new state from saved data
    new_state = state.replace(**saved_data)
    print(f"Restored state step: {new_state.step}")
    
    # Verify they produce same results
    logits1 = state(test_images, train=False)
    logits2 = new_state(test_images, train=False)
    print(f"States produce same results: {jnp.allclose(logits1, logits2)}")

if __name__ == '__main__':
    main()