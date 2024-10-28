## Experiments with `torchtitan` on Frontier

This is a copy of the [`torchtitan`](https://github.com/pytorch/torchtitan) library that I use to run LLM training experiments on Frontier. 

### Prerequisites

* Install PyTorch nightly with ROCm 6.2:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
```
My PyTorch-ROCm version is nightly `2.6.0.dev20241005+rocm6.2` and I think a reasonably recent nightly version is necessary to reproduce the results below.

* Clone this repo and install the following packages:
```bash
pip install datasets torchdata tomli tensorboard sentencepiece tiktoken blobfile tabulate ninja
``` 

* Download the Llama-3.1-8B tokenizer:

```python 
python torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original" --hf_token=...
```

where `hf_token` is your Hugging Face Hub token.

* Unlike for CUDA, you will need to install FlashAttention-2 for ROCm separately. [This page](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-acceleration-libraries.html) provides the instructions for that. Basically, to intall from source:

```bash
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention/
GPU_ARCHS=gfx90a python setup.py install  # MI200 series
```
Here, `gfx90a` is the correct GPU architecture choice for MI250X. In the last step, make sure to build with `ninja` (`pip install ninja` if it's not already installed), otherwise it might take forever. Also, make sure to set your ROCm home directory correctly for the installation to proceed: *e.g.* `export ROCM_HOME=/opt/rocm-6.2.0` for ROCm 6.2.

* Install the `aws-ofi-rccl` plugin, which enables `rccl` (AMD ROCm's version of `nccl`) to use `libfabric` for a more performant interconnect. I provide a shell script here ([`aws_ofi_rccl.sh`](https://github.com/eminorhan/frontier-torchtitan/blob/master/aws_ofi_rccl.sh)) to install this plugin. Simply run this script (*e.g.* `sh aws_ofi_rccl.sh`) to install the plugin (the script assumes that your ROCm version is 6.2.0; if you're using a different version, change it accordingly).

### Training

The SLURM batch script in [`train.sh`](https://github.com/eminorhan/frontier-torchtitan/blob/master/train.sh) can be used to train a Llama-3.1-8B model with a context size of 8192 tokens. This script uses the training config file in [`train_configs/llama3_8b.toml`](https://github.com/eminorhan/frontier-torchtitan/blob/master/train_configs/llama3_8b.toml). Feel free to modify the config according to your needs.

### A note on IP network interfaces

For loading and saving distributed checkpoints, the code uses the `torch.distributed.checkpoint` (DCP) library. A new process group with the `gloo` backend is created for this purpose (separate from the process group used by `nccl` for training). In my experience, the IP network interface to be used by both `gloo` and `nccl` needs to be explicitly set to `hsn0`, *i.e.*:
```bash
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
```
Otherwise, it becomes impossible to run on more than ~300 nodes due to communication failures.

### Results

#### Head-to-head comparison between A100 *vs.* MI250X GPUs (8 nodes)
`torchtitan` repo provides performance benchmarks for training Llama-3 8B with context size 8192 on 64 A100 GPUs (8 nodes) [here](https://github.com/pytorch/torchtitan/blob/main/docs/performance.md). With FSDP2 only parallelism, selective activation checkpointing, and a local batch size of 1, they report a `wps` of ~2900 (tokens/second) and `mfu` of ~58% (model flops utilization). 

I replicated the same set-up on 8 Frontier nodes with 64 GCDs and observed a `wps` of ~1260 and `mfu` of ~23% only. This is ~2.3-2.5x worse than the A100 results. Despite occasional [reports](https://www.databricks.com/blog/training-llms-scale-amd-mi250-gpus) I see claiming that AMD MI250X is competitive with NVIDIA A100, MI250X performs much worse than A100 on serious AI workloads in my experience. This substantial difference is likely due to the advantage in interconnect NVIDIA has over AMD with NVLink and NCCL (see also below).

#### Scaling up on Frontier
I've been able to scale Llama-3 8B training with FSDP2 + DP + TP (`dp_shard` + `dp_replicate` + `tp`) up to 640 nodes (5120 GCDs) on Frontier. With `dp_shard=32`, `dp_replicate=20`, `tp=8`, and a batch size per `dp_degree` of 21, this setup consumes a hefty 110M tokens per update globally (`32*20*21*8192`). The wall-clock time per update is around ~0.6 minute, so this setup would take around ~3.8 days to go through 1T tokens. However, this setup doesn't work reliably unfortunately. Out of 10 attempts, I would be lucky if 4-5 worked successfully. Large-scale runs on a large number of nodes are regrettably, disappointingly finicky on Frontier. Increasing the node count beyond 640 (by increasing `dp_replicate`) almost always fails in my experience.
