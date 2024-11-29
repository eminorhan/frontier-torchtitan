from datasets import load_dataset, load_from_disk, interleave_datasets
import random

def extract_code(rec):
    files = rec["files"]
    print(f"Number of files in record: {len(files)}")
    random.shuffle(files)
    if random.random() < 0.5:
        text = f"<repo_name>{rec['repo_name']}"
        for f in files:
            text += f"<file_sep>{f['path']}\n{f['text']}"
    else:
        text = ""
        for f in rec["files"]:
            text += f"<file_sep>{f['text']}"
    rec["text"] = text
    return rec

# make sure each component is iterable
ds_dclm = load_dataset("Zyphra/Zyda-2", name="dclm_crossdeduped", split="train", streaming=True)
ds_fwe = load_dataset("Zyphra/Zyda-2", name="fwe3", split="train", streaming=True).remove_columns("language_score")
ds_dolma = load_dataset("Zyphra/Zyda-2", name="dolma-cc_crossdeduped-filtered", split="train", streaming=True)
ds_zyda = load_dataset("Zyphra/Zyda-2", name="zyda_crossdeduped-filtered", split="train", streaming=True)
ds_stack = load_from_disk("/lustre/orion/stf218/scratch/emin/huggingface/stack_v2_smol").to_iterable_dataset()
ds_openwebmath = load_dataset("open-web-math/open-web-math", split="train", streaming=True)

# interleave componenets with given probabilities
ds = interleave_datasets(
    [ds_dclm, ds_fwe,ds_dolma, ds_zyda, ds_stack, ds_openwebmath], 
    probabilities=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
    seed=1, 
    stopping_strategy="all_exhausted"
    )

# print some examples
for i, example in enumerate(ds):
    if i >= 10:
        break
    if "files" in example.keys():
        example = extract_code(example)
    print(example["text"])    
    # print(example.keys())
    # print(example['repo_name'])
    print("====================")