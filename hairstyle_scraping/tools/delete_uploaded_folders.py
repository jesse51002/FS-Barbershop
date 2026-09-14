import os
import shutil
import sys
sys.path.insert(0,'./src')
import Constants


def delete_uploaded_folders(root_dir, finished_upload_file):
    finished_upload = []
    # Instanties the finihsed folders from file
    with open(finished_upload_file,'r') as file:
        for x in file.readlines():
            line = x.strip()
            finished_upload.append(line)
            
    for dir in os.listdir(root_dir):
        # Only deletes arleady uploaded files
        if dir not in finished_upload:
            continue
                
        # Deletes the uploaded file
        shutil.rmtree(os.path.join(root_dir, dir))
        
        print(f"Removed {dir}")
            
        
if __name__ == "__main__":
    
    
    chosen = -1
    while chosen < 1 or chosen > 2:
        print("""
        Delete already uploaded folder from?
        1. Raw
        2. Clean
        """)
        chosen = int(input())
        

        if chosen >= 1 or chosen <= 2:
            print(f"""
            You have picked option {chosen}, are sure this action is irreversible\n
            Type 'confirm' to proceed
            """)

            if input() != "confirm":
                print(f"'confirm' was typed incorrectly. Restart...")
                continue
        else:
            print(f"{chosen} is an invalid choice, pick a valid choice")
            
    if chosen == 1:
        delete_uploaded_folders(Constants.RAW_BODY_IMAGES_DIR, finished_upload_file=Constants.FINIHSED_BODY_RAW_UPLOAD)
    elif chosen == 2:
        delete_uploaded_folders(Constants.CLEAN_BODY_IMAGES_DIR, finished_upload_file=Constants.FINIHSED_BODY_CLEAN_UPLOAD)