from datasets import load_dataset

ds = load_dataset("HuggingFaceTB/finemath", "finemath-3plus", split="train", num_proc=32, trust_remote_code=True)
# ds = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", num_proc=32, trust_remote_code=True)
# ds = load_dataset("HuggingFaceTB/finemath", "infiwebmath-3plus", split="train", num_proc=32, trust_remote_code=True)
# ds = load_dataset("HuggingFaceTB/finemath", "infiwebmath-4plus", split="train", num_proc=32, trust_remote_code=True)

print(f"Done!")