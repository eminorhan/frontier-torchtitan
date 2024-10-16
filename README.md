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

### A note on IP network interfaces

For loading and saving distributed checkpoints, the code uses the `torch.distributed.checkpoint` (DPC) library. A new process group with the `gloo` backend is used for this purpose (separate from the process group used for training). In my experience, the IP network interface used by `gloo` for loading ans saving distributed checkpoints needs to be explicitly set to the same interface as the one used by `nccl` for training, *i.e.*:
```bash
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
```
Otherwise, it becomes impossible to run on more than ~300 nodes due to communication failures.

### Results

#### Head-to-head comparison between A100 *vs.* MI250X GPUs (8 nodes)
`torchtitan` repo provides performance benchmarks for training Llama-3 8B with context size 8192 on 64 A100 GPUs (8 nodes) [here](https://github.com/pytorch/torchtitan/blob/main/docs/performance.md). With FSDP2 only parallelism, selective activation checkpointing, and a local batch size of 1, they report a `wps` of ~2900 (tokens/second) and `mfu` of ~58% (model flops utilization). 

I replicated the same set-up on 8 Frontier nodes with 64 GCDs and observed a `wps` of ~645 and `mfu` of ~9% only. This is ~6x worse than the A100 results. Despite occasional [reports](https://www.databricks.com/blog/training-llms-scale-amd-mi250-gpus) I see claiming that AMD MI250X is competitive with NVIDIA A100, MI250X performs much worse than A100 on serious AI workloads in my experience. This substantial difference is likely due to the advantage in interconnect NVIDIA has over AMD with NVLink and NCCL (see also below).

**Update (Oct 1):** After going through the profile traces, I noticed that the code was spending an inordinate amount of time on the `bwd_kernel_dk_dv` operation. A quick Google search brought up [this isssue](https://github.com/pytorch/pytorch/issues/135431), which seems to affect recent versions of PyTorch-ROCm. Indeed, updating to the most recent nightly (`2.6.0.dev20240930+rocm6.2`) yielded a ~2-2.5x improvement in the training throughput. In the above setting (8 nodes), I'm now observing a `wps` of ~1250 and `mfu` of ~23%. Although this is a significant (and welcome) improvement over the previous version, there's still a ~2.5x gap with the A100 results. I'm currently investigating if I can reduce this gap even further.

#### Scaling up on Frontier
I've been able to scale Llama-3 8B training with FSDP2 + DP + TP (`dp_shard` + `dp_replicate` + `tp`) up to 832 nodes (6656 GCDs) on Frontier. With `dp_shard=32`, `dp_replicate=26`, `tp=8`, and a batch size per `dp_degree` of 21, this setup consumes a hefty 143M tokens per update globally (`32*26*21*8192`). The wall-clock time per update is around ~1 minute, so this setup would take around ~4.9 days to go through 1T tokens. However, this setup doesn't work reliably unfortunately. Out of 10 attempts, I would be lucky if 1-2 worked successfully. Large-scale runs on a large number of nodes are regrettably, disappointingly finicky on Frontier. Increasing the node count beyond 832 (by increasing `dp_replicate`) almost always fails in my experience.
