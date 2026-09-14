import sys
sys.path.insert(0,'./src')

import os
import cv2
import pandas as pd
import shutil

from Detection import detection_model
import Constants
from aws_s3_downloader import download_aws_folder, get_download_folders
from aws_s3_uploader import upload_aws_folder


detect_model = detection_model()
root_dir = Constants.CLEAN_BODY_IMAGES_DIR


def main():
    removed_count = 0
    total_count = 0

    dataframe = None
    if os.path.isfile(Constants.DATAFRAME_SAVE_FILE):
        dataframe = pd.read_csv(Constants.DATAFRAME_SAVE_FILE)

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

    for folder in folders_to_anaylze:
        print("Recleaning", folder)
        folder_path = os.path.join(root_dir, folder)
    
        folder_downloaded = False
        if not os.path.isdir(folder_path):
            key = rel_base + folder + ".zip"
            zip_process_thread = download_aws_folder(folder_path, key)
            zip_process_thread.join()
            folder_downloaded = True

        removed_from_folder = False
        
        for img_name in os.listdir(folder_path):
            img_pth = os.path.join(folder_path, img_name)
            if not os.path.isfile(img_pth):
                continue

            img = cv2.imread(img_pth)
            faces = detect_model.inference(img)

            keep_image = len(faces) < 2
            if not keep_image:
                print(f"{len(faces)} found, not keeping: {img_pth}", )
                os.remove(img_pth)

                if dataframe is not None:
                    relative_path = "/".join(img_pth.split("/")[-3:])
                    dataframe = dataframe.loc[dataframe["ImagePath"] != relative_path, :]
                
                removed_count += 1
                removed_from_folder = True

            total_count += 1

            if total_count % 100 == 0:
                print(f"Looked at {total_count} and removed {removed_count} images")

        if folder_downloaded:
            if removed_from_folder:
                upload_aws_folder(folder_path, os.path.join(rel_base, folder))
            shutil.rmtree(folder_path)

    if dataframe is not None:
        if os.path.isfile(Constants.DATAFRAME_SAVE_FILE):
            os.remove(Constants.DATAFRAME_SAVE_FILE)
        dataframe.to_csv(Constants.DATAFRAME_SAVE_FILE, sep=',', index=False, encoding='utf-8')

        print("UPDATED THE CSV INSIGHTS FILE")
    
    print(f"FINISHED!!! Looked at {total_count} and removed {removed_count} images")


if __name__ == "__main__":
    main()
            