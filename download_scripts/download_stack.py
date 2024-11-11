from datasets import load_dataset

ds = load_dataset("bigcode/the-stack-dedup", split="train", num_proc=32, trust_remote_code=True)
print(f"Done!")