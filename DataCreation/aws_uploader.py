import os

import boto3
from boto3.s3.transfer import TransferConfig
import shutil

# Test it on a service (yours may be different)
s3resource = boto3.client('s3')

BUCKET_NAME = "fs-upper-body-gan-dataset"


DESTROYED_DATA_ROOT = "./DestroyedData"

config = TransferConfig(multipart_threshold=1024*25, max_concurrency=10,
                        multipart_chunksize=1024*25, use_threads=True)
    
    
zip_output_location = DESTROYED_DATA_ROOT + ".zip"
    
print(f"making zip for {DESTROYED_DATA_ROOT}")
shutil.make_archive(DESTROYED_DATA_ROOT, 'zip', DESTROYED_DATA_ROOT)
print(f"finished zipping for {DESTROYED_DATA_ROOT}")
    
print(f"uploading {DESTROYED_DATA_ROOT}.zip")
s3resource.upload_file(zip_output_location, BUCKET_NAME, os.path.basename(DESTROYED_DATA_ROOT) + ".zip",
ExtraArgs={ 'ACL': 'public-read', 'ContentType': 'video/mp4'},
Config = config,
)
print(f"Finished uploading {DESTROYED_DATA_ROOT}.zip")


# delete zip file
os.remove(zip_output_location)