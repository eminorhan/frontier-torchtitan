#!/bin/bash

#SBATCH --account=stf218
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:19:00
#SBATCH --job-name=eval_ckpt
#SBATCH --output=eval_ckpt_%A_%a.out
#SBATCH --array=0
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
export HF_HUB_OFFLINE=1
# export TRUST_REMOTE_CODE=true
# export HF_DATASETS_TRUST_REMOTE_CODE=true
# export CURL_CA_BUNDLE=""

srun tune run eleuther_eval --config eval_config.yaml

echo "Done"