import boto3
from botocore import UNSIGNED
from botocore.config import Config
import gzip
from datasets import load_dataset
from botocore.exceptions import ClientError

num_proc = 100
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket_name = "softwareheritage"

def download_contents(blob_id):
    key = f"content/{blob_id}"
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        with gzip.GzipFile(fileobj=obj['Body']) as fin:
            content = fin.read().decode("utf-8", errors="ignore")
        return {"text": content, "download_success": True}
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"File not found: {key}")
            return {"text": "", "download_success": False}
        else:
            raise

ds = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", num_proc=num_proc)
ds = ds.map(download_contents, input_columns="blob_id", num_proc=num_proc)

# Filter out failed downloads
ds = ds.filter(lambda x: x['download_success'])

# Optionally, print the first example to verify the data
print(ds[0])

ds.save_to_disk('/lustre/orion/stf218/scratch/emin/huggingface/smollm-corpus-python-edu', num_proc=num_proc)