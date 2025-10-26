###############################
#
#  Minimal JAX Tutorial: CIFAR-10 Classifier with TrainState
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
import tensorflow_datasets as tfds
import tensorflow as tf

# Disable GPU for TensorFlow to avoid conflicts with JAX
tf.config.set_visible_devices([], "GPU")

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

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

# CIFAR-10 CNN model
class CIFARCNN(nn.Module):
    num_classes: int = 10
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # Input is already (batch, 32, 32, 3) for CIFAR-10
        # First conv block
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
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
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        # Dense layers
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.2, deterministic=not train)(x)
        x = nn.Dense(features=self.num_classes)(x)
        
        return x

# Training function
def train_step(state, batch_images, batch_labels):
    """Single training step using TrainState."""
    
    # fold in step to get a per-step dropout key (optional but good hygiene)
    dropout_rng = None
    if state.rng is not None:
        dropout_rng = jax.random.fold_in(state.rng, state.step)

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        # IMPORTANT: make batch_stats mutable and pass dropout rng
        logits, mutable = state.apply_fn(
            variables, batch_images, train=True,
            mutable=['batch_stats'],
            rngs={'dropout': dropout_rng} if dropout_rng is not None else None
        )
        loss = optax.softmax_cross_entropy(logits, batch_labels).mean()
        accuracy = (jnp.argmax(logits, axis=-1) == jnp.argmax(batch_labels, axis=-1)).mean()
        # return updated batch_stats via aux
        return loss, {'accuracy': accuracy, 'batch_stats': mutable['batch_stats']}

    # Take a grad step; info carries accuracy and new batch_stats
    new_state, info = state.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
    new_state = new_state.replace(batch_stats=info['batch_stats'])
    return new_state, {'accuracy': info['accuracy']}

# Evaluation function
def eval_step(state, batch_images, batch_labels):
    """Evaluation step."""
    logits = state(batch_images, train=False)
    loss = optax.softmax_cross_entropy(logits, batch_labels).mean()
    accuracy = (jnp.argmax(logits, axis=-1) == jnp.argmax(batch_labels, axis=-1)).mean()
    return {'loss': loss, 'accuracy': accuracy}

def load_cifar_data(batch_size=128):
    """Load CIFAR-10 dataset using TensorFlow Datasets."""
    
    def preprocess_fn(data):
        image = tf.cast(data['image'], tf.float32) / 255.0  # Normalize to [0, 1]
        label = data['label']
        return image, label
    
    # Load training data
    train_ds = tfds.load('cifar10', split='train', as_supervised=False)
    train_ds = train_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    train_ds = tfds.as_numpy(train_ds)
    
    # Load test data
    test_ds = tfds.load('cifar10', split='test', as_supervised=False)
    test_ds = test_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = tfds.as_numpy(test_ds)
    
    return train_ds, test_ds

def train_epoch(state, train_ds, num_batches_per_epoch=None):
    """Train for one epoch."""
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    num_batches = 0
    
    for batch_images, batch_labels in train_ds:
        # Convert labels to one-hot
        batch_labels_onehot = jax.nn.one_hot(batch_labels, 10)
        
        # Training step
        state, train_info = train_step(state, batch_images, batch_labels_onehot)
        
        # Accumulate metrics
        epoch_loss += train_info['accuracy']  # Using accuracy as proxy for loss tracking
        epoch_accuracy += train_info['accuracy']
        num_batches += 1
        
        # Limit batches per epoch if specified (for faster training)
        if num_batches_per_epoch and num_batches >= num_batches_per_epoch:
            break
    
    avg_loss = epoch_loss / num_batches
    avg_accuracy = epoch_accuracy / num_batches
    
    return state, avg_loss, avg_accuracy

def evaluate(state, test_ds, num_batches=None):
    """Evaluate on test set."""
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches_eval = 0
    
    for batch_images, batch_labels in test_ds:
        # Convert labels to one-hot
        batch_labels_onehot = jax.nn.one_hot(batch_labels, 10)
        
        # Evaluation step
        eval_info = eval_step(state, batch_images, batch_labels_onehot)
        
        # Accumulate metrics
        total_loss += eval_info['loss']
        total_accuracy += eval_info['accuracy']
        num_batches_eval += 1
        
        # Limit batches if specified
        if num_batches and num_batches_eval >= num_batches:
            break
    
    avg_loss = total_loss / num_batches_eval
    avg_accuracy = total_accuracy / num_batches_eval
    
    return avg_loss, avg_accuracy

def main():
    """Main training loop for CIFAR-10 classification."""
    print("JAX Tutorial: CIFAR-10 Classifier with TrainState")
    print("=" * 50)
    
    # Set random seed
    rng = jax.random.PRNGKey(42)
    np.random.seed(42)
    
    # Training parameters
    batch_size = 128
    num_epochs = 10  # Changed from 5 to 10 as requested
    learning_rate = 0.001
    num_classes = 10
    
    print(f"Loading CIFAR-10 dataset...")
    train_ds, test_ds = load_cifar_data(batch_size=batch_size)
    
    # Get a sample batch to initialize the model
    sample_images, sample_labels = next(iter(train_ds))
    print(f"Data shapes:")
    print(f"  Images: {sample_images.shape}")
    print(f"  Labels: {sample_labels.shape}")
    
    # Create model
    model = CIFARCNN(num_classes=num_classes)
    
    # Initialize parameters - extract both params and batch_stats
    rng, init_rng = jax.random.split(rng, 2)
    variables = model.init(init_rng, sample_images, train=True)
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    # Create optimizer
    tx = optax.adam(learning_rate=learning_rate)
    
    # Create TrainState
    state = TrainState.create(model, params, batch_stats=batch_stats, tx=tx, rng=init_rng)
    
    print(f"\nModel initialized with {sum(x.size for x in jax.tree_util.tree_leaves(params))} parameters")
    print(f"Initial step: {state.step}")
    print(f"Starting training for {num_epochs} epochs...")
    print("-" * 50)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Train for one epoch
        state, train_loss, train_acc = train_epoch(state, train_ds, num_batches_per_epoch=None)  # Limit batches for faster training
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(state, test_ds, num_batches=50)  # Limit batches for faster evaluation
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.4f}")
        print(f"  Step: {state.step}")
        print("-" * 50)
    
    print(f"\nTraining completed!")
    print(f"Final step: {state.step}")
    
    # Final evaluation on full test set
    print(f"\nFinal evaluation on full test set...")
    final_test_loss, final_test_acc = evaluate(state, test_ds)
    print(f"Final Test Loss: {final_test_loss:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.4f}")
    
    # Demonstrate inference on a few samples
    print(f"\nDemonstrating inference on sample images...")
    test_images, test_labels = next(iter(test_ds))
    sample_images = test_images[:5]
    sample_labels = test_labels[:5]
    
    # Get predictions
    logits = state(sample_images, train=False)
    predictions = jnp.argmax(logits, axis=-1)
    
    print(f"Sample predictions:")
    for i in range(5):
        print(f"  Image {i+1}: True={sample_labels[i]}, Pred={predictions[i]}")
    
    print(f"Sample accuracy: {(predictions == sample_labels).mean():.3f}")

if __name__ == '__main__':
    main()