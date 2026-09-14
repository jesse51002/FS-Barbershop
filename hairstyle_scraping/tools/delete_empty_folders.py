import sys
sys.path.insert(0,'./src')

from Utils import delete_empty
import Constants

delete_empty(Constants.ACCEPTED_BODY_IMAGES_DIR)
delete_empty(Constants.ACCEPT_BODY_BACK_REM_IMAGES_DIR)
print("Deleted empty folders in accept directory")

delete_empty(Constants.CLEAN_BODY_IMAGES_DIR)   
delete_empty(Constants.CLEAN_BODY_BACK_REM_IMAGES_DIR)
print("Deleted empty folders in clean directory")

delete_empty(Constants.RAW_BODY_IMAGES_DIR)
print("Deleted empty folders in raw directory")