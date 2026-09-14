import os
import shutil

import sys
sys.path.insert(0, './')

os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"

import boto3
from threading import Thread

data_folder = "./data"
hair_only_folder = os.path.join(data_folder, "hair_only")
full_img_folder = os.path.join(data_folder, "full_img")

# s3 boto documentation
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html

# Test it on a service (yours may be different)
s3resource = boto3.client('s3')
BUCKET_NAME = "fs-upper-body-gan-dataset"

hair_folders = ["staright", "wavy", "curly", "braids_dreads", "men", "afro"]
RAW_DOWNLOAD_ROOT_KEY = "HairTypes/"
HAIR_ONLY_ROOT_KEY = "HairTypes_HairOnly/"

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
            
                
def download_from_aws(split_dir, root_aws_key):
    rel_base = split_dir.split("/")[-1] + "/"
    
    download_folders = [root_aws_key + hair_type + ".zip" for hair_type in hair_folders]

    threads = []
    
    for key in download_folders:
        abs_folder_path = os.path.join(split_dir, key.split("/")[-1][:-4])
        
        zip_process_thread = download_aws_folder(abs_folder_path, key)
        threads.append(zip_process_thread)

    for t in threads:
        t.join()
    

if __name__ == "__main__":
    chosen = -1
    while chosen < 1 or chosen > 2:
        print("""
        Choose download mode
        1. Raw
        2. Hair
        """)
        chosen = int(input())

        if chosen >= 1 or chosen <= 2:
            print(f"""
            You have picked option {chosen}, are sure this action is irreversible\n
            Type 'confirm' to proceed
            """)

            if input() != "confirm":
                print(f"'confirm' was typed incorrectly. Restart...")
                chosen = -1
                continue
        else:
            print(f"{chosen} is an invalid choice, pick a valid choice")

    if chosen == 1:
        download_from_aws(full_img_folder, RAW_DOWNLOAD_ROOT_KEY)
    elif chosen == 2:
        download_from_aws(hair_only_folder, HAIR_ONLY_ROOT_KEY)
