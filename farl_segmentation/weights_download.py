import boto3

s3 = boto3.resource('s3')
BUCKET_NAME = "fs-upper-body-gan-dataset"

import boto3

s3 = boto3.client('s3')
s3.download_file(BUCKET_NAME, "rtnet50-fcn-14.torch", "/home/sagemaker-user/FS-MaskRestoration/farl_segmentation/ibug/face_parsing/rtnet/weights/rtnet50-fcn-14.torch")
