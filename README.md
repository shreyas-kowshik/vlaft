# VLA Fine Tuning Recipes

## Installation

Install LIBERO and activate `libero` environment

```
conda create -n libero2 python=3.10
conda activate libero2
pip install uv
uv pip install -r requirements.txt
uv pip install transformers==4.40.2
uv pip install -e .
uv pip install numpy==1.24.1

python -m pip install -U "jax[cuda12]==0.6.2" # Torch breaks a bit as numpy goes to 2.0.0, jax needs it apparently
```

From current repository
```
pip install -r requirements.txt
```

## Checkpoints

ViT: `https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view`

## LIBERO data setup

`export HF_HOME="/data/user_data/skowshik"`

`cd data/user_data/<username>/PATH_TO_DATASET`

`git clone https://huggingface.co/datasets/physical-intelligence/libero  ./libero_converted`

Dataset uses `lerobot` format in above converted dataset (HugginFace's dataset format for robot learning)

## Design

Better to use `LeRobot` format, use the huggingface dataset abstraction. Can easily push dataset to huggingface for usage across the community.


Blueprint

```
class RobotDataset
```

## Training

```
python train.py --phase pretrain --root_dir /data/user_data/skowshik/datasets/libero_pro/
```