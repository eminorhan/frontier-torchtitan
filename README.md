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

### Pretraining data
Currently, the planned pretraining data consist of a combination of the following datasets:

* [Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2), which is itself a cross-deduplicated and filtered combination of DCLM (3.3T), FineWeb-Edu (1.3T), Dolma (0.2T), Zyda (0.2T).

* `python-edu` subset of the [`smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (4B).

* [`OpenWebMath`](https://huggingface.co/datasets/open-web-math/open-web-math) (14.7B)

The subdirectory [`download_scripts`](https://github.com/eminorhan/frontier-torchtitan/tree/master/download_scripts) contains basic Python scripts to download these datasets.

### Training
The SLURM batch script in [`train_8B.sh`](https://github.com/eminorhan/frontier-torchtitan/blob/master/train_8B.sh) can be used to train a Llama-3.1-8B model with a context size of 8192 tokens. This script uses the training config file in [`train_configs/llama3_8b.toml`](https://github.com/eminorhan/frontier-torchtitan/blob/master/train_configs/llama3_8b.toml). Feel free to modify the config according to your needs.

### A note on IP network interfaces
For loading and saving distributed checkpoints, the code uses the `torch.distributed.checkpoint` (DCP) library. A new process group with the `gloo` backend is created for this purpose (separate from the process group used by `nccl` for training). In my experience, the IP network interface to be used by both `gloo` and `nccl` needs to be explicitly set to `hsn0`, *i.e.*:
```bash
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
```
Otherwise, it becomes impossible to run on more than ~300 nodes due to communication failures.

### Checkpoint conversions
Two utility scripts to convert checkpoints between `DCP` and `torch.save` formats are provided here. [`llama_to_dcp.py`](https://github.com/eminorhan/frontier-torchtitan/blob/master/llama_to_dcp.py) converts a checkpoint saved with `torch.save` to `DCP` format. This is useful when initially converting the original Llama-3 checkpoints into `DCP` format to continue pretraining them with the code in this repository (you will most likely need to use this only once before starting continued pretaining). You can do this as follows:
```bash
python llama_to_dcp.py --input_dir INPUT_DIR --ouput_dir OUTPUT_DIR
```
where `INPUT_DIR` is the directory where the original checkpoint is saved (downloaded from [here](https://huggingface.co/meta-llama/Llama-3.1-8B/tree/main/original) for the 8B model) and `OUTPUT_DIR` is the directory where the `DCP` checkpoint will be saved. The bulk of this script was copied from [this PR](https://github.com/pytorch/torchtitan/commit/3247841423429faf37bdf6918204350db293e482) by [`rlsl (Rasmus)`](https://github.com/rlrs). 

For the conversion in the other direction (`DCP --> torch.save`), you can use the [`dcp_to_llama.py`](https://github.com/eminorhan/frontier-torchtitan/blob/master/dcp_to_llama.py) script like so:
```bash
python dcp_to_llama.py --input_dir INPUT_DIR --ouput_dir OUTPUT_DIR
```
where `INPUT_DIR` now holds the `DCP` checkpoint and the `.pth` checkpoint will be saved in `OUTPUT_DIR`. You will need to do this conversion to evaluate the intermediate checkpoints. Optionally, you can also push the intermediate checkpoints (converted into `.pth` format) by passing the argument `--push_to_hub`.

### Results
#### Head-to-head comparison between A100 *vs.* MI250X GPUs (8 nodes)
`torchtitan` repo provides performance benchmarks for training Llama-3 8B with context size 8192 on 64 A100 GPUs (8 nodes) [here](https://github.com/pytorch/torchtitan/blob/main/docs/performance.md). With FSDP2 only parallelism, selective activation checkpointing, and a local batch size of 1, they report a `wps` of ~2900 (tokens/second) and `mfu` of ~58% (model flops utilization). 

I replicated the same set-up on 8 Frontier nodes with 64 GCDs and observed a `wps` of ~1300 and `mfu` of ~37% only. This is ~1.6-2.2x worse than the A100 results. Despite occasional [reports](https://www.databricks.com/blog/training-llms-scale-amd-mi250-gpus) I see claiming that AMD MI250X is competitive with NVIDIA A100, MI250X performs significantly worse than A100 on serious AI workloads in my experience. This difference is likely due to the advantage in interconnect NVIDIA has over AMD with NVLink and NCCL (see also below).

#### Scaling up on Frontier
I've been able to scale Llama-3 8B training with FSDP2 + DP + TP (`dp_shard` + `dp_replicate` + `tp`) up to 640 nodes (5120 GCDs) on Frontier. With `dp_shard=32`, `dp_replicate=20`, `tp=8`, and a batch size per `dp_degree` of 21, this setup consumes a hefty 110M tokens per update globally (`32*20*21*8192`). The wall-clock time per update is around ~0.6 minute, so this setup would take around ~3.8 days to go through 1T tokens. However, this setup doesn't work reliably unfortunately. Out of 10 attempts, I would be lucky if 4-5 worked successfully. Large-scale runs on a large number of nodes are regrettably, disappointingly finicky on Frontier. Increasing the node count beyond 640 (by increasing `dp_replicate`) almost always fails in my experience.
