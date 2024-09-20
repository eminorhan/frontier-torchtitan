## Experiments with `torchtitan` on Frontier

This is a copy of the [`torchtitan`](https://github.com/pytorch/torchtitan) library that I use to run LLM training experiments on Frontier. 

### Prerequisites

* Follow the instructions [here](https://github.com/eminorhan/frontier-accelerate) to install PyTorch-ROCm, FlashAttention-2, and the `aws-ofi-rccl` plugin. 

* Clone this repo and install the requirements here (`pip install -r requirements.txt`). 

* Download the Llama-3.1-8B tokenizer:

```python torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original" --hf_token=...```

where `hf_token` is your Hugging Face Hub token.

### Training

The SLURM batch script in [`train.sh`](https://github.com/eminorhan/frontier-torchtitan/blob/master/train.sh) can be used to train a Llama-3.1-8B model with a context size of 8192 tokens. This script uses the training config file in [`train_configs/llama3_8b.toml`](https://github.com/eminorhan/frontier-torchtitan/blob/master/train_configs/llama3_8b.toml). Feel free to modify the config according to your needs.

### Results

The training script currently uses FSDP2 + DP + TP (`dp_shard` + `dp_replicate` + `tp`). I've been able to scale this basic FSDP2 + DP + TP setup to 672 nodes (5376 GCDs) on Frontier. With `dp_shard=32`, `dp_replicate=21`, `tp=8`, and a batch size per `dp_degree` of 21, this setup consumes a hefty 116M tokens per update globally (`32*21*21*8192`). The wall-clock time per update is around ~1 minute, so this setup would take around ~6 days to go through 1T tokens.

Increasing the node count beyond 672 (by increasing `dp_replicate`) causes a failure in this setup for unknown reasons at the moment.
