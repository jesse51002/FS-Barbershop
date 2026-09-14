import os
import json
import random
import shutil
import string
import time
import boto3
from boto3.s3.transfer import TransferConfig

data_root = "./data/accepted_images"
hair_classes_json = "HairClasses.json"

target_folder = os.path.join("data", "HairTypes")

def match_hair_type(pth: str, match_list: list[str]) -> bool:
    name = os.path.basename(pth)
    for match in match_list:
        if match.lower() in name.lower():
            return True
        
    return False

def copy_selected_folders():
    """
    Copies selected folders from the data_root directory to the target_folder directory.
    The folders are organized based on the hair classes specified in the hair_classes_json file.
    Folders that match the ignore list are ignored.
    """
    # Create the target folder if it doesn't exist
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)
        
    os.makedirs(target_folder)
    
    # Load the hair classes from the json file
    with open(hair_classes_json, 'r') as f:
        hair_classes = json.load(f)
        
    # Create the subdirectories in the target folder based on the hair classes
    for hair_class in hair_classes:
        os.makedirs(os.path.join(target_folder, hair_class), exist_ok=True)

    # Iterate through the general styles in the data_root directory
    for general_style in os.listdir(data_root):
        general_pth = os.path.join(data_root, general_style)
        if not os.path.isdir(general_pth):
            continue
        
        # Iterate through the specific styles in the general style directory
        for specific_style in os.listdir(general_pth):
            specific_pth = os.path.join(general_pth, specific_style)
            if not os.path.isdir(specific_pth):
                continue
            
            # Check if the specific style should be ignored
            ignore_list = hair_classes["ignore"]
            if match_hair_type(specific_pth, ignore_list):
                print(f"{specific_pth} is ignored")
                continue
            
            found_match = False
            
            # Organize the hair classes based on whether they contain the word "men"
            classes = list(hair_classes.keys())
            classes.remove("ignore")
            
            def class_sort(x):
                if "men" in x:
                    return 0
                elif "braid" in x:
                    return 1
                else:
                    return 2
                
            
            classes = sorted(classes, key=lambda x: class_sort(x))
            
            # Iterate through the hair classes and copy the specific style if it matches
            for hair_class in classes:
                match_list = hair_classes[hair_class]
                if not match_hair_type(specific_pth, match_list):
                    continue 
                    
                # Create the subdirectory in the target folder based on the hair class and general style
                os.makedirs(os.path.join(target_folder, hair_class, general_style), exist_ok=True)
                shutil.copytree(
                    specific_pth, os.path.join(target_folder, hair_class, general_style, specific_style),
                    dirs_exist_ok=True
                    )
                found_match = True
                print(f"{specific_pth} copied to {hair_class}/{general_style}")
                break
                
                
            # Assert that a match was found for the specific style
            assert found_match, f"{specific_pth} not matched in {hair_classes_json}"
    
    # Flatten the hair class folders
    for hair_class in hair_classes:
        hair_class_pth = os.path.join(target_folder, hair_class)
        
        # Rename the files in the hair class folder
        for root, dirs, files in os.walk(hair_class_pth):
            for file in files:
                pth = os.path.join(root, file)
                
                str_time = str(time.time()).replace(".", "")
                random_chars = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
                target_file_name =  str_time + random_chars + os.path.splitext(file)[1]
                
                target_pth = os.path.join(hair_class_pth, target_file_name)
                os.rename(pth, target_pth)
        
        # Remove the hair class folder
        for root in os.listdir(hair_class_pth):
            pth = os.path.join(hair_class_pth, root)
            if os.path.isdir(pth):
                shutil.rmtree(pth)
        


                
        
def count_class_images():
    img_types = ['png', 'jpeg', 'jpg', 'webp']

    # Finds all the images in a directory
    def find_images(path: str):
        img_list = []
        
        if os.path.isdir(path):
            for sub in os.listdir(path):
                if ".ipynb_checkpoints" in sub:
                    continue
                img_list += find_images(os.path.join(path, sub))
        else:
            name = os.path.basename(path)
            file_type = name.split(".")[-1]
            if file_type in img_types:
                img_list.append(path)
        
        return img_list
                
    for texture in os.listdir(target_folder):
        folder_pth = os.path.join(target_folder, texture)
        if not os.path.isdir(folder_pth):
            continue

        print(f"{texture} has {len(find_images(folder_pth))} images")
  
def upload_to_s3():
    s3resource = boto3.client('s3',)
    BUCKET_NAME = "fs-upper-body-gan-dataset"
    
    for texture in os.listdir(target_folder):
        abs_folder_path = os.path.join(target_folder, texture)    
        rel_path = os.path.join(os.path.basename(target_folder), texture)
        
        config = TransferConfig(multipart_threshold=1024*25, max_concurrency=10,
                        multipart_chunksize=1024*25, use_threads=True)
    
        abs_folder_path = abs_folder_path.replace("\\", "/")
        rel_path = rel_path.replace("\\", "/")
        
        zip_output_location = abs_folder_path + ".zip"
        
        print(f"making zip for {rel_path}")
        shutil.make_archive(abs_folder_path, 'zip', abs_folder_path)
        print(f"finished zipping for {rel_path}")
        
        print(f"uploading {rel_path}.zip")
        s3resource.upload_file(
            zip_output_location, BUCKET_NAME, rel_path + ".zip",
            ExtraArgs={ 'ACL': 'public-read', 'ContentType': 'video/mp4'},
            Config=config,
        )
        print(f"Finished uploading {rel_path}.zip")
        
        print(f"Deleting {rel_path} from local\n\n")
        os.remove(zip_output_location)
   
if __name__ == "__main__":
    chosen = -1
    while chosen < 1 or chosen > 3:
        print("""
        Hair Type Data Handler
        1. Copy selected folders to target folder
        2. Count class images
        3. Upload to S3
        """)
        chosen = int(input())
        

        if chosen < 1 or chosen > 3:
            print(f"{chosen} is an invalid choice, pick a valid choice")
            
    if chosen == 1:
        copy_selected_folders()
    elif chosen == 2:
        count_class_images()
    elif chosen == 3:
        upload_to_s3()
    else:
        raise Exception("invalid choice")