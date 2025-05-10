#!/bin/bash
#SBATCH --gres=gpu:1 # So GPU can dung
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
python facial_verification_lvlm.py --return_result 0 --pretrained llava-next-interleave-7b --model_name llava_qwen --dataset lfw_original
