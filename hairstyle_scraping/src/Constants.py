import os

# Hair Styles
STYLES_FILES = "./styles.txt"

# Faces files
RAW_IMAGES_DIR = "./images/raw_images"
CLEAN_IMAGES_DIR = "./images/clean_images"
ACCEPTED_IMAGES_DIR = "./images/accepted_images"

FINIHSED_RAW_TXT = "./images/finished_raw"


# Body Queries
BODY_QUERIES_FILES = "./gan_dataset_tags.txt"
COUNTRIES_FILES = "./countries"

# Body files
RAW_BODY_IMAGES_DIR = "./data/raw_images"
CLEAN_BODY_IMAGES_DIR = "./data/clean_images"
CLEAN_BODY_BACK_REM_IMAGES_DIR = "./data/clean_images_background_removed"
ACCEPTED_BODY_IMAGES_DIR = "./data/accepted_images"
ACCEPT_BODY_BACK_REM_IMAGES_DIR = "./data/accepted_images_background_removed"

FINIHSED_BODY_RAW_TXT = "./data/finished_raw"
FINIHSED_BODY_RAW_UPLOAD = "./data/finished_raw_aws_upload"
FINIHSED_BODY_RAW_DOWNLOAD = "./data/finished_raw_aws_download"

FINIHSED_BODY_CLEAN_TXT = "./data/finished_clean"
FINIHSED_BODY_CLEAN_UPLOAD = "./data/finished_clean_aws_upload"
FINIHSED_BODY_CLEAN_DOWNLOAD = "./data/finished_clean_aws_download"
FINIHSED_BODY_BACK_REM_CLEAN_DOWNLOAD = "./data/finished_clean_back_rem_aws_download"
FLICKR_CREDS_FILE = "./flickr_creds"


DATAFRAME_SAVE_FILE = "./data/clean_image_insights.csv"

# Stop file
STOP_FILE = "stop.json"

SCRAPE_ADDS = ["", "front profile", "side profile"]


# Process counts
HAIR_CLEAN_PROCESSES = 1
HAIR_SCRAPE_PROCESSES = 6

BODY_CLEAN_PROCESSES = 6
BODY_SCRAPE_PROCESSES = 1

def make_dirs():
    # Makes hair importart paths
    if not os.path.exists(RAW_IMAGES_DIR):
        os.makedirs(RAW_IMAGES_DIR)
    if not os.path.exists(CLEAN_IMAGES_DIR):
        os.makedirs(CLEAN_IMAGES_DIR)
    if not os.path.exists(ACCEPTED_IMAGES_DIR):
        os.makedirs(ACCEPTED_IMAGES_DIR)
    
    
    # Makes body importart paths
    if not os.path.exists(RAW_BODY_IMAGES_DIR):
        os.makedirs(RAW_BODY_IMAGES_DIR)
    if not os.path.exists(CLEAN_BODY_IMAGES_DIR):
        os.makedirs(CLEAN_BODY_IMAGES_DIR)
    if not os.path.exists(ACCEPTED_BODY_IMAGES_DIR):
        os.makedirs(ACCEPTED_BODY_IMAGES_DIR)
        
        
  