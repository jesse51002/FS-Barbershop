import boto3

s3 = boto3.resource('s3')
BUCKET_NAME = "fs-upper-body-gan-dataset"

s3.Bucket(BUCKET_NAME).upload_file("/home/sagemaker-user/FS-Barbershop/farl_segmentation/ibug/face_parsing/rtnet/weights/rtnet50-fcn-14.torch", "rtnet50-fcn-14.torch")