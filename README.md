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

Overall, I achieved similar training throughput with `torchtitan` that I did with the HF `transformers` based code base [here](https://github.com/eminorhan/frontier-accelerate): with a global batch size of 16.8M tokens per update, each training update takes around a minute to complete on 64 Frontier nodes. The training script currently uses FSDP2 + DP (`dp_shard` + `dp_replicate`). I haven't observed any benefits to adding TP to the fold (3D parallelism). Presumably, 8B is too small to benefit from 3D parallelism. 

I've been able to scale this basic FSDP2 + DP setup to 256 nodes on Frontier. With `dp_shard=256` and `dp_replicate=8` and a batch size per `dp_degree` of 4, this setup consumes a hefty 67M tokens per update globally (`256*8*4*8192`). The wall-clock time per update is still around 1 minute (*i.e.* roughly linear scaling), so this setup would take around ~10.3 days to go through 1T tokens.
