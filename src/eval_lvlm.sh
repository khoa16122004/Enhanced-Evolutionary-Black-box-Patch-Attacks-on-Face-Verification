#!/bin/bash
#SBATCH --gres=gpu:1 # So GPU can dung
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

python facial_verification_lvlm.py --return_result 1 --pretrained llava-next-interleave-7b --model_name llava_qwen --dataset lfw
python facial_verification_lvlm.py --return_result 1 --pretrained llava-onevision-qwen2-7b-ov --model_name llava_qwen --dataset lfw
python facial_verification_lvlm.py --return_result 1 --pretrained Mantis-8B-siglip-llama3 --model_name mantis --dataset lfw
