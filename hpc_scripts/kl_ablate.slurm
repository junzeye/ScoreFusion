#!/bin/bash

#SBATCH -J lambdas
#SBATCH -p gpu
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -t 0-12:00:00     # runtime limit. max is 1 day
#SBATCH -G 4
#SBATCH -o out/logs/lambdas_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=YOUR_EMAIL

# Spawn 4 GPU jobs in parallel on different GPUs in the same node
CUDA_VISIBLE_DEVICES=0 python sdxl_kl.py 512 1 0.2 > out/logs/l_20.log 2>&1 &
echo "started 0.2 job!"
CUDA_VISIBLE_DEVICES=1 python sdxl_kl.py 512 1 0.4 > out/logs/l_40.log 2>&1 &
echo "started 0.4 job!"
CUDA_VISIBLE_DEVICES=2 python sdxl_kl.py 512 1 0.6 > out/logs/l_60.log 2>&1 &
echo "started 0.6 job!"
CUDA_VISIBLE_DEVICES=3 python sdxl_kl.py 512 1 0.8 > out/logs/l_80.log 2>&1 &
echo "started 0.8 job!"


# CUDA_VISIBLE_DEVICES=0 python sdxl_kl.py 512 1 0 > out/logs/l_0.log 2>&1 &
# echo "started 0 job!"
# CUDA_VISIBLE_DEVICES=1 python sdxl_kl.py 512 1 1.0 > out/logs/l_100.log 2>&1 &
# echo "started 1 job!"

sleep 100
echo "nvidia-smi output after 100 seconds:"
nvidia-smi

sleep 1000
echo "nvidia-smi output after 1100 seconds:"
nvidia-smi

sleep 5000
echo "nvidia-smi output after 6100 seconds:"
nvidia-smi

# wait for each job to finish, and print the nvidia-smi output
wait
echo "All jobs finished!"
nvidia-smi