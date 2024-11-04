import os
import argparse
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP to Llama.")
    parser.add_argument("--input_dir", type=Path, help="Input directory with DCP weights.")
    parser.add_argument("--output_dir", type=Path, help="Output directory for Llama weights.")
    args = parser.parse_args()

    # DCP_CKPT_DIR = "outputs/checkpoint/step-0"
    # LLAMA_CKPT_PATH = "outputs/pt/checkpoint.pth"

    llama_path = os.path.join(args.output_dir, "checkpoint.pth")

    # convert dcp model to torch.save (assumes checkpoint was generated as above)
    print(f"DCP --> torch conversion ({args.input_dir} --> {args.output_dir})")
    dcp_to_torch_save(args.input_dir, llama_path)

    print(f"Loading checkpoint torch.load")
    x = torch.load(llama_path, map_location='cpu')

    print(f"Saving model state_dict only with torch.save")
    torch.save(x["model"], llama_path)