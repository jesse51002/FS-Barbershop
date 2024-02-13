import os

import boto3
import shutil

# Test it on a service (yours may be different)
s3resource = boto3.client('s3')

BUCKET_NAME = "fs-upper-body-gan-dataset"


DESTROYED_DATA_ROOT = "./MaskData/DestroyedData"

zip_file = DESTROYED_DATA_ROOT + ".zip"



print(f"Starting {zip_file} download")
s3resource.download_file(Bucket=BUCKET_NAME, Key=os.path.basename(zip_file), Filename=zip_file)
print(f"Downloaded {zip_file}")

print(f"unzip for {DESTROYED_DATA_ROOT}")
shutil.unpack_archive(zip_file, DESTROYED_DATA_ROOT)
os.remove(zip_file)

print("Finished downloading and unzipping data")