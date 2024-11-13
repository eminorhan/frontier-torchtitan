from datasets import load_dataset

ds = load_dataset("Zyphra/Zyda-2", name="fwe3", split="train", num_proc=32, trust_remote_code=True)
print(f"Done loading!")

it = 0
for sample in ds:
    print(sample)
    if it == 10:
        break
    it += 1