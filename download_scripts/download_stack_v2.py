import boto3
import gzip
from botocore import UNSIGNED
from botocore.config import Config
from datasets import load_dataset
from botocore.exceptions import ClientError


s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
bucket_name = "softwareheritage"


def download_contents(files):
    download_success = True
    for file in files:
        try:
            key = f"content/{file['blob_id']}"
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            with gzip.GzipFile(fileobj=obj['Body']) as fin:
                file["text"] = fin.read().decode("utf-8", errors="ignore")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"File not found: {key}")    
                file["text"] = ""
                download_success = False
    return {"files": files, "download_success": download_success}


num_proc = 1000
ds = load_dataset("bigcode/the-stack-v2-train-smol-ids", split="train", num_proc=num_proc, trust_remote_code=True)
ds = ds.map(lambda row: download_contents(row["files"]), num_proc=num_proc)
ds = ds.filter(lambda x: x['download_success'], num_proc=num_proc)  # filter out failed downloads

# print the first example to verify the data
print(ds[0])

# optionally save the preprocessed data to disk
ds.save_to_disk('/lustre/orion/stf218/scratch/emin/huggingface/stack_v2_smol', num_shards=3000)
print('Done!')