#!/bin/bash

#SBATCH -J gpu_job
#SBATCH -p gpu
#SBATCH -c 12
#SBATCH -N 1
#SBATCH -t 0-05:59:00     # runtime limit. max is 1 day
#SBATCH -G 2
#SBATCH -o out/logs/sdxl_merged_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=YOUR_EMAIL

CUDA_VISIBLE_DEVICES=0 python sdxl_ckpt_merged.py 512 1 1.0 > out/logs/merged_100.log 2>&1 &
echo "started checkpoint merging 100% job!"

CUDA_VISIBLE_DEVICES=1 python sdxl_ckpt_merged.py 512 1 0.0 > out/logs/merged_0.log 2>&1 &
echo "started checkpoint merging 0% job!"

# CUDA_VISIBLE_DEVICES=0 python sdxl_ckpt_merged.py 512 1 0.2 > out/logs/merged_20.log 2>&1 &
# echo "started checkpoint merging 20% job!"

# CUDA_VISIBLE_DEVICES=1 python sdxl_ckpt_merged.py 512 1 0.4 > out/logs/merged_40.log 2>&1 &
# echo "started checkpoint merging 40% job!"

# CUDA_VISIBLE_DEVICES=2 python sdxl_ckpt_merged.py 512 1 0.6 > out/logs/merged_60.log 2>&1 &
# echo "started checkpoint merging 60% job!"

# CUDA_VISIBLE_DEVICES=3 python sdxl_ckpt_merged.py 512 1 0.8 > out/logs/merged_80.log 2>&1 &
# echo "started checkpoint merging 80% job!"

sleep 100
echo "nvidia-smi output after 100 seconds:"
nvidia-smi

sleep 1000
echo "nvidia-smi output after 1100 seconds:"
nvidia-smi

# wait for each job to finish, and print the nvidia-smi output
wait
echo "All jobs finished!"