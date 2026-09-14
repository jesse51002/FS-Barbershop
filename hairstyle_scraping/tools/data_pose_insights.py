import sys
sys.path.insert(0,'./src')

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil
import boto3

import Constants
from aws_s3_downloader import download_aws_folder, get_download_folders
import pandas as pd
import json


root_dir = Constants.ACCEPT_BODY_BACK_REM_IMAGES_DIR

REJECTED_DIR  = "data/rejected"

JSON_OUTPUT = "data/pose_insights_json.json"
CONFIDENCE = 0.5

def get_pose_acceptance(img_pth, pose_inferencer):
    img = cv2.imread(img_pth)
    
    pose = pose_inferencer.index_to_dic(pose_inferencer.inference(img)[0]["keypoints"])
    
    hand_max = max(
        pose["right_shoulder"][1] if pose["right_shoulder"][2] > CONFIDENCE else 0, 
        pose["left_shoulder"][1] if pose["left_shoulder"][2] > CONFIDENCE else 0
        )
    
    elbow_offset = 50
    hand_offset = 100
    
    hand_lowered = (
        (pose["left_elbow"][1] > hand_max + elbow_offset or pose["left_elbow"][2] < CONFIDENCE) and
        (pose["left_hand"][1] > hand_max + hand_offset or pose["left_hand"][2] < CONFIDENCE)  and
        (pose["right_elbow"][1] > hand_max + elbow_offset or pose["right_elbow"][2] < CONFIDENCE)  and
        (pose["right_hand"][1] > hand_max + hand_offset or pose["right_hand"][2] < CONFIDENCE) 
    )
    
    return hand_lowered


def create_pose_insights():
    from ViTPose.pose_extract import pose_model
    
    os.makedirs(REJECTED_DIR, exist_ok=True)
    os.makedirs(root_dir, exist_ok=True)
    
    if os.path.isfile(JSON_OUTPUT):
        with open(JSON_OUTPUT, "r") as f:
            output_dict = json.load(f)
    else:
        output_dict = {
            "accepted": [],
            "rejected": []
        }

    pose_inferencer = pose_model()

    # Gets folders already done to not use duplicates
    already_done_folders = set()
    for file in output_dict["accepted"]:
        folder_name = file.split("/")[-2]
        already_done_folders.add(folder_name)
    for file in output_dict["rejected"]:
        folder_name = file.split("/")[-2]
        already_done_folders.add(folder_name)
    
    rel_base = root_dir.split("/")[-1] + "/"

    folders_to_anaylze = []
    
    # Adds folders on local
    folders_to_anaylze += os.listdir(root_dir)
    
    # Adds folders in AWS S3
    s3_folders = get_download_folders(rel_base)
    for i in range(len(s3_folders)):
        s3_folders[i] = s3_folders[i].split("/")[-1][:-4]
    folders_to_anaylze += s3_folders

    # Removes duplicates
    folders_to_anaylze = set(folders_to_anaylze) - already_done_folders
    
    total_images_done = len(output_dict["accepted"]) + len(output_dict["rejected"])
    for folder in folders_to_anaylze:
        print("Getting insights from", folder)
        folder_path = os.path.join(root_dir, folder)
        folder_downloaded = False
        if not os.path.isdir(folder_path):
            key = rel_base + folder + ".zip"
            zip_process_thread = download_aws_folder(folder_path, key)
            zip_process_thread.join()
            folder_downloaded = True

        rejected_list = []
        
        for img_name in os.listdir(folder_path):
            img_pth = os.path.join(folder_path, img_name)
            accepted = get_pose_acceptance(img_pth, pose_inferencer)
            
            save_pth = "/".join(img_pth.replace("\\", "/").split("/")[-2:])
            if accepted:
                output_dict["accepted"].append(save_pth)
            else:
                rejected_list.append(img_pth)
                output_dict["rejected"].append(save_pth)
                
            total_images_done += 1
            print(f"{total_images_done}: {accepted} from {save_pth}")
    
        if folder_downloaded:
            shutil.rmtree(folder_path)

        with open(JSON_OUTPUT, "w") as f:
            print("Accepted: ", len(output_dict["accepted"]), "\nRejected: ", len(output_dict["rejected"]))
            json.dump(output_dict, f, indent=2)
            
        for img_pth in rejected_list:
            rej_pth = os.path.join(REJECTED_DIR, os.path.basename(img_pth))
            if os.path.isfile(rej_pth):
                os.remove(rej_pth)
            os.rename(img_pth, rej_pth)

if __name__ == "__main__":
    create_pose_insights()
    