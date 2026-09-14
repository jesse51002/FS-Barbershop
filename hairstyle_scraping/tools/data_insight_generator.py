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

ANGLE_FACE_MULT = 2

VISUALIZATION_SAVE_DIRECTORY = "./data/clean_insights_visualization"
S3_INSIGHTS_KEY = "clean_image_insights.csv"
S3_INSIGHTS_BUCKET = "fs-upper-body-gan-dataset"

COLUMNS = ["ImagePath", "looking_forward", "HairType", "Ethnicity", "Sex", "HandDown"]

FRONT_ANGLES = np.array([[0, 13], [0, 13], [0, 13]])

root_dir = Constants.CLEAN_BODY_IMAGES_DIR


def get_face_angle(img, detect_model, angle_model):
    faces = detect_model.inference(img)

    # No face detected
    if len(faces) == 0:
        print("No faces detected... Rejected")
        return None, None, None
        
    x, y, w, h = faces[0]

    mid_x, mid_y = int(x + w/2), int(y + h/2)
    max_size = int(max(w, h) / 2)
    scale_size = max_size * ANGLE_FACE_MULT
    
    bounds = (
            mid_x - scale_size,  # left
            mid_x + scale_size,   # right
            mid_y - scale_size,  # bottom
            mid_y + scale_size  # top
    )
        
    l, r, b, t = bounds
        
    l_bounds, r_bounds, b_bounds, t_bounds = (
        max(0,-1*l), 
        max(0,r - img.shape[1]),
        max(0,-1*b),
        max(0,t - img.shape[0]),
    )

    bounded_image = img[
            max(0, b): min(img.shape[0], t),
            max(0, l): min(img.shape[1], r)
            ]
        
    bounded_image = cv2.copyMakeBorder(
        bounded_image,
        b_bounds,  # bottom
        t_bounds,  # top
        l_bounds,   # left
        r_bounds,  # right
        cv2.BORDER_CONSTANT  # borderType
        )

    pitch, yaw, roll = angle_model.predict(bounded_image)
    return pitch, yaw, roll


def get_image_insights(img_pth, clip_model, pose_inferencer, detect_model, angle_model):
    img = cv2.imread(img_pth)

    if img is None:
        return pd.DataFrame({
            "ImagePath": None,
            "looking_forward": None,
            "HairType": None,
            "Ethnicity": None,
            "Sex": None,
            "HandDown": None,
        }, index=[0])
        
    relative_path = "/".join(img_pth.split("/")[-3:])

    pitch, yaw, roll = get_face_angle(img, detect_model, angle_model)
    looking_forward = (
        (abs(pitch) > FRONT_ANGLES[0, 0] and abs(pitch) < FRONT_ANGLES[0, 1]) and
        (abs(yaw) > FRONT_ANGLES[1, 0] and abs(yaw) < FRONT_ANGLES[1, 1]) and
        (abs(roll) > FRONT_ANGLES[2, 0] and abs(roll) < FRONT_ANGLES[2, 1])
    ) if pitch is not None else None

    classifications = clip_model.inference(img)
    for key in classifications:
        classifications[key] = classifications[key].argmax()
    
    classifications["HairType"] = clip_model.hair_types_list[classifications["HairType"]]
    classifications["Ethnicity"] = clip_model.ethnicites_list[classifications["Ethnicity"]]
    classifications["Sex"] = clip_model.sex_list[classifications["Sex"]]
    
    pose = pose_inferencer.index_to_dic(pose_inferencer.inference(img)[0]["keypoints"])

    hand_lowered = (
        pose["left_elbow"][1] > pose["left_shoulder"][1] and
        pose["left_hand"][1] > pose["left_elbow"][1] and
        pose["right_elbow"][1] > pose["right_shoulder"][1] and
        pose["right_hand"][1] > pose["right_elbow"][1]
    )
    
    img_insights = pd.DataFrame({
        "ImagePath": relative_path,
        "looking_forward": looking_forward,
        "HairType": classifications["HairType"],
        "Ethnicity": classifications["Ethnicity"],
        "Sex": classifications["Sex"],
        "HandDown": hand_lowered
    }, index=[0])
    
    return img_insights


def create_insights_pandas():
    from Detection import detection_model
    from Clip import DataClassifierClip
    from ViTPose.pose_extract import pose_model
    from sixdrepnet import SixDRepNet
    
    if os.path.isfile(Constants.DATAFRAME_SAVE_FILE):
        dataframe = pd.read_csv(Constants.DATAFRAME_SAVE_FILE)
        print("Loading csv from save file")
    else:
        dataframe = pd.DataFrame(columns=COLUMNS)
    
    angle_model = SixDRepNet()
    detect_model = detection_model()
    clip_model = DataClassifierClip()
    pose_inferencer = pose_model()

    # Gets folders already done to not use duplicates
    already_done_folders = set()
    for i in range(dataframe.shape[0]):
        if pd.isna(dataframe.loc[i, "ImagePath"]):
            continue
        folder_name = dataframe.loc[i, "ImagePath"].split("/")[-2]
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
    
    total_images_done = 0
    for folder in folders_to_anaylze:
        print("Getting insights from", folder)
        folder_path = os.path.join(root_dir, folder)
        folder_downloaded = False
        if not os.path.isdir(folder_path):
            key = rel_base + folder + ".zip"
            zip_process_thread = download_aws_folder(folder_path, key)
            zip_process_thread.join()
            folder_downloaded = True

        for img_name in os.listdir(folder_path):
            img_pth = os.path.join(folder_path, img_name)
            img_insights = get_image_insights(img_pth, clip_model, pose_inferencer, detect_model, angle_model)
            dataframe = pd.concat([dataframe, img_insights], ignore_index=True)
            total_images_done += 1
            print(f"{total_images_done}: got insights from {img_pth}")
    
        if folder_downloaded:
            shutil.rmtree(folder_path)

    # Saves the dataframe
    dataframe.to_csv(Constants.DATAFRAME_SAVE_FILE, sep=',', index=False, encoding='utf-8')


def create_visualizations():
    print("Create visualizations")
    
    if not os.path.isdir(VISUALIZATION_SAVE_DIRECTORY):
        os.makedirs(VISUALIZATION_SAVE_DIRECTORY)

    dataframe = pd.read_csv(Constants.DATAFRAME_SAVE_FILE)

    for col in dataframe.columns:
        if col == "ImagePath":
            continue
            
        # make the plot
        dataframe.groupby([col]).count().plot(kind='pie', autopct='%1.0f%%', title=col, y='ImagePath')
        # save the plot
        plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, col))
        print("Saved visualizations for", col)


def export_insights():
    s3resource = boto3.client('s3')
    s3resource.upload_file(Constants.DATAFRAME_SAVE_FILE, S3_INSIGHTS_BUCKET, S3_INSIGHTS_KEY)
    print("Uploaded insights csv")


def download_insights():
    s3resource = boto3.client('s3')
    if os.path.isfile(Constants.DATAFRAME_SAVE_FILE):
        print("This will override current dataframe insights file\n Type 'confirm' to continue")
        user_input = input()
        if user_input != "confirm":
            exit()
        os.remove(Constants.DATAFRAME_SAVE_FILE)
        print("Deleted old dataframe")
        
    s3resource.download_file(Bucket=S3_INSIGHTS_BUCKET, Key=S3_INSIGHTS_KEY, Filename=Constants.DATAFRAME_SAVE_FILE)
    print("Downloaded insights csv")


if __name__ == "__main__":
    chosen = -1
    while chosen < 1 or chosen > 4:
        print("""
        Do you want to create csv file or visualize
            1. Create csv
            2. Visualize csv
            3. Export csv
            4. Download csv
            """)
        chosen = int(input())

    if chosen == 1:
        create_insights_pandas()
    elif chosen == 2:
        create_visualizations()
    elif chosen == 3:
        export_insights()
    elif chosen == 4:
        download_insights()
    