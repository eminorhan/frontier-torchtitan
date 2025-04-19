#!/bin/bash

#SBATCH --account=stf218
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=eval_ckpt
#SBATCH --output=eval_ckpt_%A_%a.out
#SBATCH --array=8
#SBATCH --qos=debug

# set proxy server to enable communication with outside
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

# set misc env vars
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/lustre/orion/stf218/scratch/emin/huggingface"
export HF_DATASETS_CACHE="/lustre/orion/stf218/scratch/emin/huggingface"
# export HF_HUB_OFFLINE=1

# irectory where eval config files are stored
CONFIG_DIR="/lustre/orion/stf218/scratch/emin/torchtitan/eval_configs"

# list of eval configs to run
CONFIG_FILES=(
  "config_mmlu.yaml"
  "config_mmlu_pro.yaml"
  "config_arc.yaml"
  "config_gsm8k.yaml"
  "config_agieval.yaml"
  "config_drop.yaml"
  "config_bbh.yaml"
  "config_winogrande.yaml"
  "config_hellaswag.yaml"
)

# suse SLURM_ARRAY_TASK_ID to select the config file based on the index
CONFIG_FILE="${CONFIG_DIR}/${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}"

srun tune run eleuther_eval --config $CONFIG_FILE

echo "Done"