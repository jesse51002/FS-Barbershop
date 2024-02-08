import os

import boto3
from boto3.s3.transfer import TransferConfig
import shutil

# Test it on a service (yours may be different)
s3resource = boto3.client('s3')

BUCKET_NAME = "fs-upper-body-gan-dataset"


DESTROYED_DATA_ROOT = "./DestroyedData"


