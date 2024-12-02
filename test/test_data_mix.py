from datasets import load_dataset, load_from_disk, interleave_datasets

def extract_code(rec):
    text = ""
    for f in rec["files"]:
        text += f"\n\n{f["text"]}"
    return text

# make sure each component is iterable
ds_dclm = load_dataset("Zyphra/Zyda-2", name="dclm_crossdeduped", split="train", streaming=True)
ds_fwe = load_dataset("Zyphra/Zyda-2", name="fwe3", split="train", streaming=True).remove_columns("language_score")  # remove `language_score` column due to dtype mismatch with dclm
ds_dolma = load_dataset("Zyphra/Zyda-2", name="dolma-cc_crossdeduped-filtered", split="train", streaming=True)
ds_zyda = load_dataset("Zyphra/Zyda-2", name="zyda_crossdeduped-filtered", split="train", streaming=True)
ds_stack = load_from_disk("/lustre/orion/stf218/scratch/emin/huggingface/stack_v2_smol").to_iterable_dataset()
ds_openwebmath = load_dataset("open-web-math/open-web-math", split="train", streaming=True)

# interleave componenets with given probabilities
ds = interleave_datasets(
    [ds_dclm, ds_fwe,ds_dolma, ds_zyda, ds_stack, ds_openwebmath], 
    probabilities=[0.425, 0.425, 0.03, 0.02, 0.095, 0.005], 
    seed=1, 
    stopping_strategy="all_exhausted"
    )

# print some examples
for i, example in enumerate(ds.skip(1000000)):
    if i >= 100:
        break
    # if example["files"] is None:
    #     sample_text = example["text"]
    # else:
    #     sample_text = extract_code(example)  # handle code
    print(example["id"])    
    # print(example.keys())
    # print(example['repo_name'])
    print("====================")