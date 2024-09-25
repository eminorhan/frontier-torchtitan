## Experiments with `torchtitan` on Frontier

This is a copy of the [`torchtitan`](https://github.com/pytorch/torchtitan) library that I use to run LLM training experiments on Frontier. 

### Prerequisites

* Follow the instructions [here](https://github.com/eminorhan/frontier-accelerate) to install PyTorch-ROCm, FlashAttention-2, and the `aws-ofi-rccl` plugin. 

* Clone this repo and install the requirements here (`pip install -r requirements.txt`). 

* Download the Llama-3.1-8B tokenizer:

```python 
python torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original" --hf_token=...
```

where `hf_token` is your Hugging Face Hub token.

### Training

The SLURM batch script in [`train.sh`](https://github.com/eminorhan/frontier-torchtitan/blob/master/train.sh) can be used to train a Llama-3.1-8B model with a context size of 8192 tokens. This script uses the training config file in [`train_configs/llama3_8b.toml`](https://github.com/eminorhan/frontier-torchtitan/blob/master/train_configs/llama3_8b.toml). Feel free to modify the config according to your needs.

### Results

#### Head-to-head comparison between A100 *vs.* MI250X GPUs (8 nodes)
`torchtitan` repo provides performance benchmarks for training Llama-3 8B with context size 8192 on 64 A100 GPUs (8 nodes) [here](https://github.com/pytorch/torchtitan/blob/main/docs/performance.md). With FSDP2 only parallelism, selective activation checkpointing, and a local batch size of 1, they report a `wps` of ~2900 and `mfu` of ~58%. I replicated the same set-up on 8 Frontier nodes with 64 GCDs. I observed a `wps` of ~645 and `mfu` of ~9% only. This is ~6x worse than the A100 results. Despite occasional [reports](https://www.databricks.com/blog/training-llms-scale-amd-mi250-gpus) I see claiming that AMD MI250X is competitive with NVIDIA A100, MI250X performs much worse than A100 on serious AI workloads in my experience. This substantial difference is likely due to the advantage in interconnect NVIDIA has over AMD with NVLink and NCCL (see also below).


#### Scaling up on Frontier
I've been able to scale Llama-3 8B training with FSDP2 + DP + TP (`dp_shard` + `dp_replicate` + `tp`) up to 832 nodes (6656 GCDs) on Frontier. With `dp_shard=32`, `dp_replicate=26`, `tp=8`, and a batch size per `dp_degree` of 21, this setup consumes a hefty 143M tokens per update globally (`32*26*21*8192`). The wall-clock time per update is around ~1 minute, so this setup would take around ~4.9 days to go through 1T tokens. However, this setup doesn't work reliably unfortunately. Out of 10 attempts, I would be lucky if 1-2 worked successfully. Large-scale runs on a large number of nodes are regrettably, disappointingly finicky on Frontier. Increasing the node count beyond 832 (by increasing `dp_replicate`) almost always fails in my experience.
