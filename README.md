# VLA Fine Tuning Recipes

## Installation

Install LIBERO and activate `libero` environment

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