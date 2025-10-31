# VLA Fine Tuning Recipes

## Installation

```
conda create -n libero python=3.10
conda activate libero
pip install uv
```

# Install PARL
Install PARL and activate `parl` environment
`uv pip install tensorflow_datasets`
`uv pip install torch`
`uv pip install -r libero_req.txt`
`pip install openai-clip`

This seems good to start training

<!-- Install LIBERO and activate `libero` environment
```
# Comment out transformers #
cd LIBERO/
uv pip install -r requirements.txt
uv pip install -e .
``` -->

<!-- Now install stuff for current repository
```
uv pip install transformers==4.40.2
uv pip install numpy==1.24.1
uv pip install "jax[cuda12]==0.6.2" # Torch breaks a bit as numpy goes to 2.0.0, jax needs it apparently
uv pip install tensorflow_cpu
uv pip install tensorflow_datasets
uv pip install protobuf==6.33.0
uv pip install numpy==1.26.4
```

```
uv pip install rlds[tensorflow]
``` -->

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
python train_jit.py --phase pretrain --learning_rate 0.0001 --root_dir /data/user_data/skowshik/datasets/libero_pro --sequence_length 13
```


Resume from checkpoint
```
python train_jit.py --phase pretrain --learning_rate 0.0001 --root_dir /data/user_data/skowshik/datasets/libero_pro --sequence_length 13 --resume_from_checkpoint /data/user_data/skowshik/datasets/libero_pro/checkpoints/jit_20251028_024223/epoch_7/
```

## Evaluation

```
python -m pdb eval_libero.py --libero_eval_max_steps 360 --finetune_type libero_10 --libero_img_size 224 --libero_path /home/skowshik/vla/codebase/envs/LIBERO --sequence_length 10 --action_pred_steps 3 --phase finetune
```