import math
import os
import shutil

import boto3

import sys
sys.path.insert(0, './src')
import Constants


MAX_IMAGES_IN_FOLDER = 5000
ROOT = Constants.RAW_BODY_IMAGES_DIR
FINSIHED_TXT_FILE = Constants.FINIHSED_BODY_RAW_TXT

already_done = []
with open(FINSIHED_TXT_FILE, 'r') as file:
    for x in file.readlines():
        line = x.strip()
        already_done.append(line)

for folder in os.listdir(ROOT):
    folder_pth = os.path.join(ROOT, folder)
    
    if not os.path.isdir(folder_pth):
        continue
    
    if folder not in already_done:
        print(f"Still scraping {folder}... Skipping")
        continue
    
    folder_size = len(os.listdir(folder_pth))
    
    if folder_size < MAX_IMAGES_IN_FOLDER:
        print(f"Skipping {folder} because has {folder_size}")
        continue
    
    folder_count = math.ceil(folder_size / MAX_IMAGES_IN_FOLDER)
    folder_amount = math.ceil(folder_size / folder_count)

    print(f"Splitting {folder}")
    
    for i, img_name in enumerate(os.listdir(folder_pth)):
        folder_number = math.floor(i / folder_amount)
        new_folder_name = f"{folder}_{folder_number}" 
        new_folder_pth = os.path.join(ROOT, new_folder_name)
        os.makedirs(new_folder_pth, exist_ok=True)
        
        original_pth = os.path.join(folder_pth, img_name)
        new_pth = os.path.join(new_folder_pth, img_name)
        
        os.rename(original_pth, new_pth)
        
    os.rmdir(folder_pth)
    
    with open(FINSIHED_TXT_FILE, 'a') as finished_file:
        for i in range(folder_count):
            finished_file.write(f'\n{folder}_{i}')
    
        
        
        
    
        