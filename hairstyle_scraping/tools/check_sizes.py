import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"
import sys
sys.path.insert(0, './src')

import cv2
import boto3
import shutil

import json
import Constants
from aws_s3_downloader import download_aws_folder, get_download_folders

root_dir = Constants.ACCEPTED_BODY_IMAGES_DIR

AWS_CLEAN_ROOT_KEY = Constants.ACCEPTED_BODY_IMAGES_DIR.split("/")[-1] + "/"

s3resource = boto3.client('s3')
BUCKET_NAME = "fs-upper-body-gan-dataset"
JSON_OUTPUT = "data/size_insights_json.json"


def check_sizes():
    if os.path.isfile(JSON_OUTPUT):
        with open(JSON_OUTPUT, "r") as f:
            output_dict = json.load(f)
    else:
        output_dict = {
            "accepted": [],
            "rejected": []
        }

    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    
    rel_base = root_dir.split("/")[-1] + "/"

    # Adds folders in AWS S3
    s3_folders = get_download_folders(rel_base)
    for i in range(len(s3_folders)):
        s3_folders[i] = s3_folders[i].split("/")[-1][:-4]
    folders_to_anaylze = s3_folders

    total_imgs = 0
    total_1024 = 0
    total_size = 0

    for folder_num, folder in enumerate(folders_to_anaylze):
        folder_path = os.path.join(root_dir, folder)

        key = rel_base + folder + ".zip"
        zip_process_thread = download_aws_folder(folder_path, key)
        zip_process_thread.join()

        for img_name in os.listdir(folder_path):
            if not img_name.endswith(".jpg"):
                continue
                
            img_pth = os.path.join(folder_path, img_name)
            
            img = cv2.imread(img_pth)

            save_pth = "/".join(img_pth.replace("\\", "/").split("/")[-2:])

            total_size += img.shape[0]
            
            if img.shape[0] >= 1024:
                total_1024 += 1
                output_dict["accepted"].append(save_pth)
            else:
                output_dict["rejected"].append(save_pth)

            total_imgs += 1

        with open(JSON_OUTPUT, "w") as f:
            perc = total_1024 / total_imgs
            average_size = total_size / total_imgs
            print(
                "Accepted: ", len(output_dict["accepted"]),
                "\nRejected: ", len(output_dict["rejected"]),
                "\nPercentage:", perc,
                "\nAverage Size:", average_size
            )
            json.dump(output_dict, f, indent=2)
        

    
if __name__ == "__main__":
    check_sizes()