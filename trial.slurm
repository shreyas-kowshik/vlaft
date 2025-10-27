#!/bin/bash
#SBATCH --job-name=seer_eval_v1
#SBATCH --partition=general          # partition name
#SBATCH --qos=normal            # qos
#SBATCH --gres=gpu:A6000:4               # request 1 GPU
#SBATCH --cpus-per-task=12          # number of CPUs
#SBATCH --time=48:00:00          # time (1 day)
#SBATCH --mem-per-cpu=4G
#SBATCH --output=outcome_%j.out      # Standard output file (%j is job ID)
#SBATCH --error=outcome_%j.err       # Standard error file (%j is job ID)

conda init
conda activate libero2
python train.py --phase pretrain --learning_rate 0.0001 --root_dir /data/user_data/skowshik/datasets/libero_pro/ --sequence_length 13 --batch_size 1

