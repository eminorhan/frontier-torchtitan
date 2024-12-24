from datasets import load_dataset, load_from_disk, interleave_datasets

def extract_code(rec):
    text = ""
    for f in rec["files"]:
        text += f"\n\n{f["text"]}"
    return text

# make sure each component is iterable
ds_dclm = load_dataset("Zyphra/Zyda-2", name="dclm_crossdeduped", split="train", streaming=True).select_columns("text")
ds_fwe = load_dataset("Zyphra/Zyda-2", name="fwe3", split="train", streaming=True).select_columns("text")
ds_dolma = load_dataset("Zyphra/Zyda-2", name="dolma-cc_crossdeduped-filtered", split="train", streaming=True).select_columns("text")
ds_zyda = load_dataset("Zyphra/Zyda-2", name="zyda_crossdeduped-filtered", split="train", streaming=True).select_columns("text")
# ds_stack = load_from_disk("/lustre/orion/stf218/scratch/emin/huggingface/stack_v2_smol").to_iterable_dataset(num_shards=3000)
ds_finemath = load_dataset("HuggingFaceTB/finemath", name="finemath-3plus", split="train", streaming=True).select_columns("text")

# interleave componenets with given probabilities
ds = interleave_datasets(
    [ds_dclm, ds_fwe,ds_dolma, ds_zyda, ds_finemath], 
    probabilities=[0.425, 0.425, 0.03, 0.02, 0.1], 
    seed=1, 
    stopping_strategy="all_exhausted"
    )

print(f"interleaved iterable dataset n_shards: {ds.n_shards}")

# print some examples
for i, example in enumerate(ds.skip(1000000)):
    if i >= 100:
        break
    # if example["files"] is None:
    #     sample_text = example["text"]
    # else:
    #     sample_text = extract_code(example)  # handle code
    print(example)
    # print(example.keys())
    # print(example['repo_name'])
    print("====================")