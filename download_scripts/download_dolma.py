from datasets import load_dataset

ds = load_dataset("Zyphra/Zyda-2", name="dolma-cc_crossdeduped-filtered", split="train", num_proc=32, trust_remote_code=True)
print(f"Done!")