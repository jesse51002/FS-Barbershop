import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"

from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role, Session
import boto3


sagemaker_session = Session(boto3.session.Session(region_name='us-east-1'))
sagemaker_execution_role = get_execution_role(sagemaker_session=sagemaker_session)

pytorch_estimator = PyTorch(
    entry_point='train.py',
    source_dir=".",
    role=sagemaker_execution_role,
    instance_type='ml.g5.12xlarge',
    instance_count=1,
    framework_version='2.3.0',
    py_version='py311',
    max_run=431999,
    hyperparameters={
        "gpus": 4,
        "cfg": "fusionstyles",
        "aug": "noaug",
        "mirror": True,
        "snap": 50,
        "upload_images_to_s3": None,
        "outdir": "This_is_not_used",
        "aws_checkpoint_name": "model_run5_3600kimgs.pkl"
    },
    metric_definitions=[
        {'Name': 'ada_aug_p:error', 'Regex': 'augment: (.*?);'},
        {'Name': 'fid:error', 'Regex': 'fid50k_full: (.*?);'},
        {'Name': 'kimg:error', 'Regex': 'kimg: (.*?);'},
    ]
)


pytorch_estimator.fit({'train': 's3://fs-upper-body-gan-dataset/accepted_images_background_removed/'})

# python train.py --gpus 4 --cfg fusionstyles --augpipe noise --mirror True --upload_images_to_s3 --outdir training_output --data ./data --snap 1 --aws_checkpoint_name start.pkl