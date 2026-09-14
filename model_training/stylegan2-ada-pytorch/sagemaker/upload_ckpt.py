import boto3
import os
import tarfile 


s3resource = boto3.client('s3')
BUCKET_NAME = "fs-upper-body-gan-dataset"

local_file = "checkpoints/network-snapshot-003600.pkl"
target_key = "training/model_run5_3600kimgs.pkl"

s3resource.upload_file(local_file, BUCKET_NAME, target_key)
print(f"Uploaded {local_file} to {target_key}")

