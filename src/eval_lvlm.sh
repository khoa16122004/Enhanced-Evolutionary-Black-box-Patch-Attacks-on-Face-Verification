#!/bin/bash
#SBATCH --gres=gpu:1 # So GPU can dung
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1


python evaluate_lvlm.py --pretrained Mantis-8B-clip-llama3 --model_name mantis --dataset lfw
python evaluate_lvlm.py --pretrained Mantis-8B-siglip-llama3 --model_name mantis --dataset lfw
