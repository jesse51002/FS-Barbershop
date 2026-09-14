import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"

import shutil

import boto3

import sys
sys.path.insert(0, './src')
import Constants

from threading import Thread, Lock
from s3_list import list_s3_from_root

# s3 boto documentation
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html

# Test it on a service (yours may be different)
s3resource = boto3.client('s3')
BUCKET_NAME = "fs-upper-body-gan-dataset"
MAX_FOLDER_COUNT_DOWNLOAD = 10000000


def zip_parser(abs_folder_path, zip_file, finished_download_file=None, file_lock=None):
    if not os.path.isdir(abs_folder_path):
        os.makedirs(abs_folder_path)

    shutil.unpack_archive(zip_file, abs_folder_path)
    os.remove(zip_file)

    if finished_download_file is not None:
        if file_lock is not None:
            file_lock.acquire()

        with open(finished_download_file, 'a') as finished_file:
            folder_name = abs_folder_path.replace("\\", "/").split("/")[-1]
            finished_file.write(f'\n{folder_name}')

        if file_lock is not None:
            file_lock.release()

    print(f"Finsihed parsing {abs_folder_path}")


def download_aws_folder(abs_folder_path, s3_key, finished_download_file=None, file_lock=None):
    if not os.path.isdir(abs_folder_path):
        os.makedirs(abs_folder_path)
    
    zip_file_pth = abs_folder_path+".zip"

    print(f"Starting {s3_key} download")
    s3resource.download_file(Bucket=BUCKET_NAME, Key=s3_key, Filename=zip_file_pth)
    print(f"Downloaded {s3_key}")

    print(f"Starting zip parsing for {s3_key}")
    zip_process_thread = Thread(target=zip_parser, args=(abs_folder_path, zip_file_pth, finished_download_file, file_lock))
    zip_process_thread.start()

    return zip_process_thread
    

def get_download_folders(prefix="", completed_download_file=None):
    keys_in_s3 = list_s3_from_root(prefix)
    if prefix in keys_in_s3:
        keys_in_s3.remove(prefix)
    
    # Gets folders that have already been downloaded
    finished_download = []
    if completed_download_file is not None:
        # Creates file if it doesnt exist
        if not os.path.isfile(completed_download_file):
            file = open(completed_download_file, 'w')
            file.close()
             
        # Instanties the finihsed folders from file
        with open(completed_download_file, 'r') as file:
            for x in file.readlines():
                line = x.strip()
                if len(line) == 0:
                    continue
                finished_download.append(line.replace("\\", "/").split("/")[-1])

    i = 0
    while i < len(keys_in_s3):
        key = keys_in_s3[i].split("/")[-1][:-4]
        for finished in finished_download:
            if finished == key:
                keys_in_s3.pop(i)
                i -= 1
                break
        i += 1

    return keys_in_s3[:min(MAX_FOLDER_COUNT_DOWNLOAD, len(keys_in_s3))]
            
                
def download_from_aws(split_dir, completed_download_file=None):
    # Creates file if it doesnt exist
    if completed_download_file is not None and not os.path.isfile(completed_download_file):
        file = open(completed_download_file, 'w')
        file.close()

    rel_base = split_dir.split("/")[-1] + "/"
    
    download_folders = get_download_folders(rel_base, completed_download_file)
    print(download_folders)
    
    threads = []

    file_lock = Lock()
    
    for key in download_folders:
        abs_folder_path = os.path.join(split_dir, key[len(rel_base):-4])
        
        zip_process_thread = download_aws_folder(abs_folder_path, key, completed_download_file, file_lock)
        threads.append(zip_process_thread)

    for t in threads:
        t.join()
    

if __name__ == "__main__":
    chosen = -1
    while chosen < 1 or chosen > 3:
        print("""
        Choose download mode
        1. Raw
        2. Clean
        3. Accepted
        """)
        chosen = int(input())

        if chosen >= 1 or chosen <= 3:
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
        download_from_aws(Constants.RAW_BODY_IMAGES_DIR, completed_download_file=Constants.FINIHSED_BODY_RAW_DOWNLOAD)
    elif chosen == 2:
        download_from_aws(Constants.CLEAN_BODY_IMAGES_DIR, completed_download_file=Constants.FINIHSED_BODY_CLEAN_DOWNLOAD)
        download_from_aws(Constants.CLEAN_BODY_BACK_REM_IMAGES_DIR, completed_download_file=Constants.FINIHSED_BODY_BACK_REM_CLEAN_DOWNLOAD)
    elif chosen == 3:
        download_from_aws(Constants.ACCEPT_BODY_BACK_REM_IMAGES_DIR)
    else:
        raise Exception("Invalid choice")
    