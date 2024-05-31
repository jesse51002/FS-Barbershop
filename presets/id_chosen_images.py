import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"
import boto3

import cv2
import hashlib


# Test it on a service (yours may be different)
s3resource = boto3.client('s3')
BUCKET_NAME = "fusion-styles-data"

NO_ID_IMAGES_DIR = "presets/preset_data/no_id_images"
RAW_PRESETS_IMAGES_DIR = "presets/preset_data/unprocessed"


S3_RAW_IMG_PRESETS_KEY = "hairstyle_presets/presets_raw_images/"

S3_MODEL_DATA_PRESETS_KEY = "hairstyle_presets/presests_model_outputs/"
s3resource = boto3.client('s3')
BUCKET_NAME = "fusion-styles-data"


def download_uploaded_raw_images():
    response = s3resource.list_objects(
        Bucket=BUCKET_NAME,
        Prefix=S3_RAW_IMG_PRESETS_KEY,
        MaxKeys=10000  # s3 list_object limit will probably be reached before this
    )

    # Parses the response to get the names
    keys = list(x["Key"] for x in response["Contents"])
    # Remove empty names (these come from folder keys)
    if S3_RAW_IMG_PRESETS_KEY in keys:
        keys.remove(S3_RAW_IMG_PRESETS_KEY)

    for key in keys:
        basename = os.path.basename(key)

        # downloads the file
        s3resource.download_file(Bucket=BUCKET_NAME, Key=key, Filename=os.path.join(RAW_PRESETS_IMAGES_DIR, basename))


def upload_presets():
    uploaded_imgs = download_uploaded_raw_images()

    # hashes and gets the next id
    max_id = -1
    image_hashes = []
    for name in os.listdir(RAW_PRESETS_IMAGES_DIR):
        # Gets the number
        num = int(name.split(".")[0])
        max_id = max(max_id, num)

        with open(os.path.join(RAW_PRESETS_IMAGES_DIR, name), 'rb') as f:
            digest = hashlib.sha1(f.read()).digest()
        image_hashes.append(digest)

    
    for img_name in os.listdir(NO_ID_IMAGES_DIR):
        if img_name in uploaded_imgs:
            print(img_name, "was already uploaded")
            continue
        
        img_pth = os.path.join(RAW_PRESETS_IMAGES_DIR, img_name)

        if not os.path.isfile(img_pth):
            continue

        print("Uploading:", img_name)
        
        target_key = S3_RAW_IMG_PRESETS_KEY + img_name
        s3resource.upload_file(img_pth, BUCKET_NAME, target_key)

        print("Finished Uploading:", img_name)

        
if __name__ == "__main__":
    upload_presets()
        