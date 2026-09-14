import boto3
import os
import tarfile 


s3resource = boto3.client('s3')
BUCKET_NAME = "sagemaker-us-east-1-259645229668"

MODEL_KEY = "output/model.tar.gz"

ROOT_S3_KEY = "pytorch-training-2024-07-25-04-33-29-210/"

DOWNLOAD_PTH = "checkpoints/"

os.makedirs(DOWNLOAD_PTH, exist_ok=True)

local_pth = os.path.join(DOWNLOAD_PTH, os.path.basename(MODEL_KEY))
s3_key = ROOT_S3_KEY + MODEL_KEY

print(f"Starting {s3_key} download")
s3resource.download_file(Bucket=BUCKET_NAME, Key=s3_key, Filename=local_pth)
print(f"Downloaded {s3_key}")


  
# open file 
file = tarfile.open(local_pth) 
  
# extracting file 
file.extractall(DOWNLOAD_PTH) 
  
file.close() 

os.remove(local_pth)
