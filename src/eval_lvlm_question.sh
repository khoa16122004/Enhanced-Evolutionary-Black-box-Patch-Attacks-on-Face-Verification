#!/bin/bash
#SBATCH --gres=gpu:1 # So GPU can dung
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

python facial_verification_question_eval_.py --lvlm_pretrained llava-onevision-qwen2-7b-ov --lvlm_model_name llava_qwen
python facial_verification_question_eval_.py --lvlm_pretrained llava-next-interleave-7b --lvlm_model_name llava_qwen
