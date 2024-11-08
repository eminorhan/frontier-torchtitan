import os
import argparse
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP to Llama.")
    parser.add_argument("--input_dir", type=Path, help="Input directory with DCP weights.")
    parser.add_argument("--output_dir", type=Path, help="Output directory for Llama weights.")
    parser.add_argument('--push_to_hub', action='store_true', help='whether to push llama ckpt to hf hub (default: false)')
    args = parser.parse_args()

    # DCP_CKPT_DIR = "outputs/checkpoint/step-0"  # input
    # LLAMA_CKPT_DIR = "outputs/pt"  # output

    llama_path = os.path.join(args.output_dir, "checkpoint.pth")

    # convert dcp model to torch.save
    print(f"\033[91mDCP --> torch conversion \033[92m({args.input_dir} --> {args.output_dir})\033[0m")
    dcp_to_torch_save(args.input_dir, llama_path)

    print(f"\033[91mLoading checkpoint torch.load\033[0m")
    x = torch.load(llama_path, map_location='cpu')

    print(f"\033[91mSaving model state_dict only with torch.save\033[0m")
    torch.save(x["model"], llama_path)

    if args.push_to_hub:
        print(f"\033[91mPushing converted ckpt to hf hub\033[0m")

        from huggingface_hub import HfApi

        api = HfApi()

        api.upload_folder(
            folder_path=args.output_dir,
            repo_id="eminorhan/smoky-llama",
            path_in_repo=args.input_dir.name,
            repo_type="model",
            token=True
        )