from datasets import load_dataset, load_from_disk, interleave_datasets

# make sure each component is iterable
ds_dclm = load_dataset("Zyphra/Zyda-2", name="dclm_crossdeduped", split="train", streaming=True)
ds_fwe = load_dataset("Zyphra/Zyda-2", name="fwe3", split="train", streaming=True).remove_columns("language_score")
ds_dolma = load_dataset("Zyphra/Zyda-2", name="dolma-cc_crossdeduped-filtered", split="train", streaming=True)
ds_zyda = load_dataset("Zyphra/Zyda-2", name="zyda_crossdeduped-filtered", split="train", streaming=True)
ds_pyedu = load_from_disk("/lustre/orion/stf218/scratch/emin/huggingface/smollm-corpus-python-edu").to_iterable_dataset()
ds_openwebmath = load_dataset("open-web-math/open-web-math", split="train", streaming=True)

# interleave componenets with given probabilities
ds = interleave_datasets(
    [ds_dclm, ds_fwe,ds_dolma, ds_zyda, ds_pyedu, ds_openwebmath], 
    probabilities=[0.01, 0.01, 0.01, 0.01, 0.95, 0.01], 
    seed=1, 
    stopping_strategy="all_exhausted"
    )

# print some examples
for i, example in enumerate(ds):
    if i >= 100:
        break
    print(example["text"])    
    # print(example.keys())
    # print(example['repo_name'])
    print("====================")    
