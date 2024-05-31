import os
import time
import sagemaker

os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"
import boto3

import json

ENDPOINT_NAME = "cosmentic-gan-endpoint-2024-04-03-17-39-08"

predictor = sagemaker.predictor.Predictor(endpoint_name=ENDPOINT_NAME)

print("Starting prediction")
start = time.time()

sagemaker_runtime = boto3.client("runtime.sagemaker")
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps(
        {
            "ImageLink": "https://input-image-for-gan-model.s3.us-east-2.amazonaws.com/test/test.png",
            "ModifiedSegmentationLink": "https://input-image-for-gan-model.s3.us-east-2.amazonaws.com/test/test.png",
            "Modifiers": {}
        })
)

response = json.loads(response["Body"].read().decode())

print("Response:", response)
print("Finished prediction in", time.time() - start)