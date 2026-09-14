import os
import sys
sys.path.insert(0,'./src')

import Constants

while True:
    print("Input the accepted image you want to delete or type 'exit' to stop")
    delete = input()
    if delete == "exit":
        break
    
    for dir in os.listdir(Constants.ACCEPTED_BODY_IMAGES_DIR):
        
        img_pth = os.path.join(Constants.ACCEPTED_BODY_IMAGES_DIR, dir, delete)
        
        if os.path.isfile(img_pth):
            os.remove(img_pth)
            print("Removed base:", img_pth)
            
    for dir in os.listdir(Constants.ACCEPT_BODY_BACK_REM_IMAGES_DIR):
        
        img_pth = os.path.join(Constants.ACCEPT_BODY_BACK_REM_IMAGES_DIR, dir, delete)
        
        if os.path.isfile(img_pth):
            os.remove(img_pth)
            print("Removed background free:", img_pth)
    
    