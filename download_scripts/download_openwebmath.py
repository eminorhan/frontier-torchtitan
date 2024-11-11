from datasets import load_dataset

ds = load_dataset("open-web-math/open-web-math", split="train", num_proc=32, trust_remote_code=True)
print(f"Done!")