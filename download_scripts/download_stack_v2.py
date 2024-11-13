import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from smart_open import open
from datasets import load_dataset


s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

def download_contents(files):
    for file in files:
        s3_url = f"s3://softwareheritage/content/{file['blob_id']}"
        with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
            file["content"] = fin.read().decode(file["src_encoding"])
    
    return {"files": files}

num_cpus = os.cpu_count()
print(f"cpu count: {num_cpus}")
ds = load_dataset("bigcode/the-stack-v2-train-full-ids", split="train", num_proc=num_cpus, trust_remote_code=True)
ds = ds.map(lambda row: download_contents(row["files"]), num_proc=num_cpus)
for row in ds:
    for file in row["files"]:
        print(file["content"])
    break

ds.save_to_disk('/lustre/orion/stf218/scratch/emin/huggingface', num_proc=num_cpus)
