import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"
import boto3
import sagemaker
from datetime import datetime

sagemaker_client = boto3.client(service_name="sagemaker")
role = sagemaker.get_execution_role()

current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
model_name = f"cosmetic-gan-{current_datetime}"

primary_container = {
    "Image": "259645229668.dkr.ecr.us-east-1.amazonaws.com/sagemaker-studio:latest",
    "ModelDataUrl": "s3://cosmetic-surgery-endpoints/cosmetic_endpoint_tars/2024-04-26-06-02-29/ai_vad_model.tar.gz"
}

create_model_response = sagemaker_client.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    PrimaryContainer=primary_container
)

endpoint_config_name = f"cosmetic-gan-config-{current_datetime}"
sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[{
        "InstanceType": "ml.g5.xlarge",
        "InitialVariantWeight": 1,
        "InitialInstanceCount": 1,
        "ModelName": model_name,
        "VariantName": "AllTraffic"}]
)

endpoint_name = f"cosmentic-gan-endpoint-{current_datetime}"
sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)


print("Deploying endpoint:", endpoint_name)