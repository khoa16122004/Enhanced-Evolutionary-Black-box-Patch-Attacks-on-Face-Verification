#!/bin/bash
#SBATCH --gres=gpu:1 # So GPU can dung
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

python facial_verification_lvlm.py --return_result 1 --pretrained deepseek-vl-7b-chat --model_name deepseek --dataset lfw