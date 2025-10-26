import requests
import json
from PIL import Image
import jax
import jax.numpy as jnp
import flaxmodels as fm


response = requests.get('https://cdn.pixabay.com/photo/2013/05/29/22/25/elephant-114543_960_720.jpg')
with open('example.jpg', 'wb') as f:
    f.write(response.content)

response = requests.get('https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json')
with open('labels.json', 'w') as f:
    f.write(response.text)

labels = json.load(open('labels.json'))

key = jax.random.PRNGKey(0)

# Load image
img = Image.open('example.jpg').resize((480, 360))

# Image should be in range [0, 1]
x = jnp.array(img, dtype=jnp.float32) / 255.0
# Add batch dimension
x = jnp.expand_dims(x, axis=0)

# Init network
resnet18 = fm.ResNet18(output='logits', pretrained='imagenet')
params = resnet18.init(key, x)

# Compute predictions
out = resnet18.apply(params, x, train=False)

# Get top 5 classes
_, top5_classes = jax.lax.top_k(out, k=5)
top5_classes = jnp.squeeze(top5_classes, axis=0)

for i in range(top5_classes.shape[0]):
    print(f'{i + 1}.', labels[top5_classes[i]])

from PIL import Image
import jax
import jax.numpy as jnp
import flaxmodels as fm

key = jax.random.PRNGKey(0)

# Load image
img = Image.open('example.jpg')
# Image should be in range [0, 1]
x = jnp.array(img, dtype=jnp.float32) / 255.0
# Add batch dimension
x = jnp.expand_dims(x, axis=0)

resnet18 = fm.ResNet18(output='activations', pretrained='imagenet')
params = resnet18.init(key, x)
# Dictionary
out = resnet18.apply(params, x, train=False)

for key in out.keys():
    print(key, out[key].shape)