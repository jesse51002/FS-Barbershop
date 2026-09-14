import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"
import json
import boto3
from threading import Thread
import shutil

s3resource = boto3.client('s3')
BUCKET_NAME = "fs-upper-body-gan-dataset"

ROOT_S3_KEY = "accepted_images_background_removed/"

FOLDER_LIST_JSON = "sagemaker/folder_list.json"

ROOT_DATA_FOLDER = "data/accept_images_background_removed/"

def zip_parser(abs_folder_path, zip_file):
    if not os.path.isdir(abs_folder_path):
        os.makedirs(abs_folder_path)

    shutil.unpack_archive(zip_file, abs_folder_path)
    os.remove(zip_file)

    print(f"Finsihed parsing {abs_folder_path}")

def download_aws_folder(abs_folder_path, s3_key):
    if not os.path.isdir(abs_folder_path):
        os.makedirs(abs_folder_path)
    
    zip_file_pth = abs_folder_path+".zip"

    print(f"Starting {s3_key} download")
    s3resource.download_file(Bucket=BUCKET_NAME, Key=s3_key, Filename=zip_file_pth)
    print(f"Downloaded {s3_key}")

    print(f"Starting zip parsing for {s3_key}")
    zip_process_thread = Thread(target=zip_parser, args=(abs_folder_path, zip_file_pth))
    zip_process_thread.start()

    return zip_process_thread


def download_folder_list():
    print("Starting folder list download")
    
    with open(FOLDER_LIST_JSON) as f:
        folder_list = json.load(f)

    for folder in folder_list:
        download_aws_folder(ROOT_DATA_FOLDER+folder, ROOT_S3_KEY+folder+".zip")
    
if __name__ == "__main__":
    download_folder_list()