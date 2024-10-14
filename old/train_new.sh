#!/bin/bash

#SBATCH --account=stf218
#SBATCH --nodes=32
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --job-name=train_llama
#SBATCH --output=train_llama_%A_%a.out
#SBATCH --array=0
#SBATCH --qos=debug

# set proxy server to enable communication with outside
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

export LOGLEVEL=INFO
export LD_LIBRARY_PATH=/lustre/orion/stf218/scratch/emin/aws-ofi-rccl/lib:$LD_LIBRARY_PATH  # enable aws-ofi-rccl
export NCCL_NET_GDR_LEVEL=3   # Can improve performance, but remove this setting if you encounter a hang/crash.
export NCCL_ALGO=TREE         # May see performance difference with either setting. (should not need to use this, but can try)
export NCCL_CROSS_NIC=1       # On large systems, this NCCL setting has been found to improve performance
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_IB_TIMEOUT=31
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCHELASTIC_ENABLE_FILE_TIMER=1
export OMP_NUM_THREADS=1
export HF_HOME="/lustre/orion/stf218/scratch/emin/huggingface"
export HF_DATASETS_CACHE="/lustre/orion/stf218/scratch/emin/huggingface"
export HF_HUB_OFFLINE=1
export GPUS_PER_NODE=8

# set network
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3442

# Calculate global rank and world size
export WORLD_SIZE=256
export RANK=$SLURM_PROCID
export LOCAL_RANK=$((SLURM_PROCID % 8))

# Optionally print the environment variables for debugging
echo $RANK
echo $LOCAL_RANK
echo $WORLD_SIZE

CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_8b.toml"}

srun --nodes=$SLURM_NNODES --cpus-per-task=7 --ntasks-per-node=8 --gpus-per-task=1 --gpu-bind=closest python -u train.py --job.config_file ${CONFIG_FILE}

echo "Done"