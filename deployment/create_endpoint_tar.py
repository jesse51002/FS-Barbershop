import os
import tarfile
import boto3
from datetime import datetime
import time

TAR_OUTPUT_NAME = "./deployment/test_model.tar.gz"
WEIGHTS_FOLDER = "./pretrained_models"

BUCKET_NAME = "cosmetic-surgery-endpoints"

S3_ENDPOINT_ROOT = "cosmetic_endpoint_tars"


print("CREATING TAR...")

start = time.time()
with tarfile.open(TAR_OUTPUT_NAME, "w:gz") as file:
    file.add("inference.py", arcname="inference.py")

    for weight_file in os.listdir(WEIGHTS_FOLDER):
        file.add(os.path.join(WEIGHTS_FOLDER, weight_file), arcname=weight_file)


print("FINISHED CREATING TAR IN", time.time() - start, "\n")

start = time.time()

print("UPLOADING TO S3...")
current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
s3_key = f"{S3_ENDPOINT_ROOT}/{current_datetime}/ai_vad_model.tar.gz"
s3 = boto3.resource('s3')
s3.meta.client.upload_file(TAR_OUTPUT_NAME, BUCKET_NAME, s3_key)

print("FINSIHED UPLOADING TO S3 in", time.time() - start, "\n")

os.remove(TAR_OUTPUT_NAME)
print("REMOVED TAR FILE FROM LOCAL")


final_output = f"""
Bucket: {BUCKET_NAME}
S3 Key: {s3_key}
URI: s3://{BUCKET_NAME}/{s3_key}
"""

print(final_output)




