import os
import json


data_folder = "data/accept_images_background_removed"

folder_list = []

for folder in os.listdir(data_folder):

    if not os.path.isdir(os.path.join(data_folder, folder)):
        continue
    
    folder_list.append(folder)
    

with open("sagemaker/folder_list.json", "w") as f:
    json.dump(folder_list, f)
    
    