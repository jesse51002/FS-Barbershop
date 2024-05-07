import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"
import boto3


# Test it on a service (yours may be different)
s3resource = boto3.client('s3')
BUCKET_NAME = "fusion-styles-data"

RAW_PRESETS_IMAGES_DIR = "unprocessed"

S3_RAW_IMG_PRESETS_KEY = "hairstyle_presets/presets_raw_images/"


def list_uploaded_images():
    response = s3resource.list_objects(
        Bucket=BUCKET_NAME,
        Prefix=S3_RAW_IMG_PRESETS_KEY,
        MaxKeys=10000  # s3 list_object limit will probably be reached before this
    )

    # Parses the response to get the names
    names = list(set([os.path.basename(x["Key"]) for x in response["Contents"]]))
    # Remove empty names (these come from folder keys)
    if '' in names:
        names.remove('')
    return names


def upload_presets():
    uploaded_imgs = list_uploaded_images()
    
    for img_name in os.listdir(RAW_PRESETS_IMAGES_DIR):
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
        