from datasets import load_from_disk

# make sure each component is iterable
ds_stack = load_from_disk("/lustre/orion/stf218/scratch/emin/huggingface/stack_v2_smol")
print('loaded dataset from disk')
ds_stack.push_to_hub("eminorhan/svs", num_shards=3000, token=True)
print('pushed dataset to hub')
