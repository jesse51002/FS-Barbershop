import os
import sys
sys.path.insert(0, './src')

import cv2
import numpy as np
import torch
import boto3

import shutil
from p3m_matting.inference import remove_background
import Constants
from aws_s3_downloader import download_aws_folder, get_download_folders
from aws_s3_uploader import upload_aws_folder

from ViTPose.pose_extract import pose_model
from Clip import Clip

TARGET_BACKGROUND_DIR = Constants.CLEAN_BODY_BACK_REM_IMAGES_DIR
BATCH_SIZE = 4
TARGET_SIZE = 1024
root_dir = Constants.CLEAN_BODY_IMAGES_DIR

CONFIDENCE = 0.5

AWS_CLEAN_ROOT_KEY = Constants.CLEAN_BODY_BACK_REM_IMAGES_DIR.split("/")[-1] + "/"

s3resource = boto3.client('s3')
BUCKET_NAME = "fs-upper-body-gan-dataset"

REJECTED_DIR = "data/rejected"


def remove_background_from_folder(folder_path, back_rem_dir):
    names = os.listdir(folder_path)
    image_pths = [os.path.join(folder_path, x) for x in names]
        
    for i in range(0, len(image_pths), BATCH_SIZE):
    
        pths = image_pths[i:min(i + BATCH_SIZE, len(image_pths))]
        batched_input = np.zeros((len(pths), TARGET_SIZE, TARGET_SIZE, 3))
    
        for j in range(len(pths)):
            img = cv2.imread(pths[j])
            if img is None:
                img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3))
            img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_CUBIC)
            batched_input[j] = img
                
        batched_input = batched_input[:, :, :, ::-1].copy()
            
        unscaled_imgs = torch.tensor(batched_input).permute((0, 3, 1, 2))
        unscaled_imgs = unscaled_imgs.float().to("cuda:0")
        img = unscaled_imgs / 255
    
        output_imgs, pred_segments = remove_background(img)
        output_imgs = output_imgs.transpose(0, 2, 3, 1)
            
        for j in range(len(pths)):
            output_img = output_imgs[j]
            name = os.path.basename(pths[j]).split(".")[0] + ".jpg"
            cv2.imwrite(os.path.join(back_rem_dir, name), output_img)

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

def remove_img(
    folder_path,
    back_rem_dir,
    relative_pth,
    send_to_rejected_dir=False
    ):
    img_pth = os.path.join(folder_path, relative_pth)
    back_rem_img_pth = os.path.join(back_rem_dir, relative_pth)
    
    if not send_to_rejected_dir:
        os.remove(back_rem_img_pth)
    else:
        os.makedirs(REJECTED_DIR, exist_ok=True)
        
        rejected_pth = os.path.join(REJECTED_DIR, os.path.basename(back_rem_img_pth))
        os.rename(back_rem_img_pth, rejected_pth)
        
    os.remove(img_pth)
    

def remove_bad_images(
    folder_path, 
    back_rem_dir, 
    pose_inferencer,
    clip_model: Clip
    ):
    total = 0
    removed = 0
    for img_name in os.listdir(back_rem_dir):
        total += 1
        back_rem_img_pth = os.path.join(back_rem_dir, img_name)
        pose_accepted = get_pose_acceptance(back_rem_img_pth, pose_inferencer)
        realtive_pth = "/".join(back_rem_img_pth.replace("\\", "/").split("/")[-2:])
        
        if not pose_accepted:
            remove_img(folder_path, back_rem_dir, realtive_pth)
            print(f"Bad pose... Rejected")
            removed += 1
            continue
        
        raw_img = cv2.imread(back_rem_img_pth)
        probs_dic = clip_model.inference(raw_img, only_quaulity=True)
        clip_accepted = probs_dic["Quality"][0, 0] >= 0.5
        
        if not clip_accepted:
            remove_img(folder_path, back_rem_dir, realtive_pth, send_to_rejected_dir=True)
            print(f"Clip rejected based on quality... Rejected")
            removed += 1
            continue
    
    percentage = removed / total
    
    print(f"Finsihed removed bad poses.\nRemvoed {removed}/{total} = {percentage}")
    

def remove_images_background():
    if not os.path.isdir(TARGET_BACKGROUND_DIR):
        os.makedirs(TARGET_BACKGROUND_DIR)
    
    pose_inferencer = pose_model()
    clip_model = Clip()
    
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
    folders_to_anaylze = set(folders_to_anaylze)

    already_done = get_download_folders(AWS_CLEAN_ROOT_KEY)
    for i in range(len(already_done)):
        already_done[i] = already_done[i].split("/")[-1][:-4]
    folders_to_anaylze -= set(already_done)

    folders_to_anaylze = list(folders_to_anaylze)
    for folder_num, folder in enumerate(folders_to_anaylze):
        folder_path = os.path.join(root_dir, folder)

        back_rem_key = os.path.join(AWS_CLEAN_ROOT_KEY, folder)

        try:
            s3resource.get_object(Bucket=BUCKET_NAME, Key=back_rem_key + ".zip")
            print(back_rem_key, "already exists")
            continue
        except Exception as e:
            print("Removing background from", folder)
        
        folder_downloaded = False
        if not os.path.isdir(folder_path):
            key = rel_base + folder + ".zip"
            zip_process_thread = download_aws_folder(folder_path, key)
            zip_process_thread.join()
            folder_downloaded = True

        back_rem_dir = os.path.join(TARGET_BACKGROUND_DIR, folder)
        if not os.path.isdir(back_rem_dir):
            os.makedirs(back_rem_dir)
        
        remove_background_from_folder(folder_path, back_rem_dir)
        remove_bad_images(folder_path, back_rem_dir, pose_inferencer, clip_model)

        if folder_downloaded:
            upload_aws_folder(folder_path, key)
            shutil.rmtree(folder_path)

        upload_aws_folder(back_rem_dir, back_rem_key)
        shutil.rmtree(back_rem_dir)
            
        print(f"Finished doing folder {folder_num + 1}/{len(folders_to_anaylze)}")


if __name__ == "__main__":
    remove_images_background()