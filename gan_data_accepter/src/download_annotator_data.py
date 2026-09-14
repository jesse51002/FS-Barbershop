import os
import json
from aws_s3_downloader import download_aws_folder

ANNOTATOR_FOLDER = "labeler_data"
DATA_FOLDER = os.path.join(ANNOTATOR_FOLDER, "data")

CLEAN_IMG_KEY_BASE = "clean_images/"
CLEAN_IMG_BACK_REM_KEY_BASE = "clean_images_background_removed/"


def download_folder(clean_folder_pth, clean_folder_back_rem_pth):
    attempt_number = 1
    downloaded = False
    while not downloaded and attempt_number <= 3:
        try:
            zip_thread = download_aws_folder(
                clean_folder_pth,
                CLEAN_IMG_KEY_BASE + os.path.basename(clean_folder_pth) + ".zip"
            )
            
            zip_thread_back_rem = download_aws_folder(
                clean_folder_back_rem_pth,
                CLEAN_IMG_BACK_REM_KEY_BASE + os.path.basename(clean_folder_pth) + ".zip"
            )
            
            downloaded = True
        except Exception as e:
            print(e, "\n When attempting to download")
            attempt_number += 1
            
    assert downloaded, "Failed attempt to download after 3 attempts"
    
    zip_thread.join()
    zip_thread_back_rem.join()


def combine_annotations():
    accepted_pths = set()
    rejected_pths = set()
    for name in os.listdir(ANNOTATOR_FOLDER):
        pth = os.path.join(ANNOTATOR_FOLDER, name)
        if not os.path.isfile(pth):
            continue
        
        with open(pth, mode="r") as f:
            json_data = json.load(f)
            
        for reject_pth in json_data["rejected"]:
            rejected_pths.add(reject_pth)
        
        for accept_pth in json_data["accepted"]:
            accepted_pths.add(accept_pth)
    
    
    accepted_pths -= rejected_pths

    print("Accepted:", len(accepted_pths))
    print("Rejected:", len(rejected_pths))

    return list(accepted_pths), list(rejected_pths)
        
    
def download_annotator_data():
    folder_name = "full"
    
    FILE_ERROR_COUNT = 0
    
    data_folder = os.path.join(DATA_FOLDER, folder_name)
    os.makedirs(data_folder, exist_ok=True)
        
    accept_folder = os.path.join(data_folder, "accepted")
    reject_folder = os.path.join(data_folder, "rejected")
    clean_folder = os.path.join(data_folder, "clean")
    os.makedirs(accept_folder, exist_ok=True)
    os.makedirs(reject_folder, exist_ok=True)
    os.makedirs(clean_folder, exist_ok=True)
    
    clean_img_back_rem = os.path.join(data_folder, "clean_images_background_removed")
    accept_img_back_rem = os.path.join(data_folder, "accept_images_background_removed")
    reject_img_back_rem = os.path.join(data_folder, "reject_images_background_removed")
    os.makedirs(clean_img_back_rem, exist_ok=True)
    os.makedirs(accept_img_back_rem, exist_ok=True)
    os.makedirs(reject_img_back_rem, exist_ok=True)
    
    accepted_pths, rejected_pths = combine_annotations()
    accepted_pths.sort()
    rejected_pths.sort()
    
    for img_rel_pth in accepted_pths:
        folder_name = img_rel_pth.split("/")[0]
        img_name = img_rel_pth.split("/")[1]
        
        accept_img_folder = os.path.join(accept_folder, folder_name)
        accept_img_folder_back_rem_pth = os.path.join(accept_img_back_rem, folder_name)
        
        accept_img_pth = os.path.join(accept_img_folder, img_name)
        accept_img_back_rem_pth = os.path.join(accept_img_folder_back_rem_pth, img_name)
        if os.path.isfile(accept_img_pth):
            print(f"Already have {accept_img_pth} skipping...")
            continue
        
        clean_folder_pth = os.path.join(clean_folder, folder_name)
        clean_folder_back_rem_pth = os.path.join(clean_img_back_rem, folder_name)
        # Downloads folder if it doesnt exist
        if not os.path.isdir(clean_folder_pth):
            download_folder(clean_folder_pth, clean_folder_back_rem_pth)
            
        clean_img_pth = os.path.join(clean_folder_pth, img_name)
        clean_img_back_rem_pth = os.path.join(clean_folder_back_rem_pth, img_name)
        
        # assert os.path.isfile(clean_img_pth), f"Clean file must exist for {clean_img_pth}"
        if not os.path.isfile(clean_img_pth):
            FILE_ERROR_COUNT += 1
            print(f"FILE_ERROR_COUNT {FILE_ERROR_COUNT}: WARNING... ERROR. Clean file must exist for {clean_img_pth}")
            continue
        
        os.makedirs(accept_img_folder, exist_ok=True)
        os.makedirs(accept_img_folder_back_rem_pth, exist_ok=True)
        os.rename(clean_img_pth, accept_img_pth)
        
        if os.path.isfile(clean_img_back_rem_pth):
            os.rename(clean_img_back_rem_pth, accept_img_back_rem_pth)
            
        
        print(f"Moved to {accept_img_pth}")
        
    for img_rel_pth in rejected_pths:
        folder_name = img_rel_pth.split("/")[0]
        img_name = img_rel_pth.split("/")[1]
        
        reject_img_folder = os.path.join(reject_folder, folder_name)
        reject_img_folder_back_rem_pth = os.path.join(reject_img_back_rem, folder_name)
        
        reject_img_pth = os.path.join(reject_img_folder, img_name)
        reject_img_back_rem_pth = os.path.join(reject_img_folder_back_rem_pth, img_name)
        if os.path.isfile(reject_img_pth):
            print(f"Already have {reject_img_pth} skipping...")
            continue
        
        clean_folder_pth = os.path.join(clean_folder, folder_name)
        clean_folder_back_rem_pth = os.path.join(clean_img_back_rem, folder_name)
        # Downloads folder if it doesnt exist
        if not os.path.isdir(clean_folder_pth):
            download_folder(clean_folder_pth, clean_folder_back_rem_pth)
            
        clean_img_pth = os.path.join(clean_folder_pth, img_name)
        clean_img_back_rem_pth = os.path.join(clean_folder_back_rem_pth, img_name)
        
        # assert os.path.isfile(clean_img_pth), f"Clean file must exist for {clean_img_pth}"
        if not os.path.isfile(clean_img_pth):
            FILE_ERROR_COUNT += 1
            print(f"FILE_ERROR_COUNT {FILE_ERROR_COUNT}: WARNING... ERROR. Clean file must exist for {clean_img_pth}")
            continue
        
        os.makedirs(reject_img_folder, exist_ok=True)
        os.makedirs(reject_img_folder_back_rem_pth, exist_ok=True)
        os.rename(clean_img_pth, reject_img_pth)
        
        if os.path.isfile(clean_img_back_rem_pth):
            os.rename(clean_img_back_rem_pth, reject_img_back_rem_pth)
            
        
        print(f"Moved to {reject_img_pth}")
    
    print("FILE_ERROR_COUNT:", FILE_ERROR_COUNT)    
    print("Finished!!!")
    
            
        

if __name__ == "__main__":
    download_annotator_data()