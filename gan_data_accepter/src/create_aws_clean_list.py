import os
from s3_list import list_s3_from_root
import boto3
import datetime

EXPORT_FILE_NAME = "aws_folder_keys_list"

def create_aws_clean_list():
    # Test it on a service (yours may be different)
    s3resource = boto3.client('s3',
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"] 
    )
    BUCKET_NAME = "fs-upper-body-gan-dataset"
    
    # return ["clean_images/adventure portrait canon.zip"]
    prefix = "clean_images/"
    keys_in_s3 = list_s3_from_root(prefix, "data/clean_images/")
    if prefix in keys_in_s3:
        keys_in_s3.remove(prefix)
    if prefix + ".ipynb_checkpoints.zip" in keys_in_s3:
        keys_in_s3.remove(prefix + ".ipynb_checkpoints.zip")
    # keys_in_s3.sort()

    date_dict = {}
    
    for i, key in enumerate(keys_in_s3):
        obj_info = s3resource.get_object(Bucket=BUCKET_NAME, Key=key)
        last_modifed_date = obj_info["LastModified"]
        date_dict[key] = datetime.datetime.timestamp(last_modifed_date)
        print("Finished", key, f"{i}/{len(keys_in_s3)}")
    
    date_sorted_keys = sorted(date_dict.items(), key=lambda x:x[1])
    
    print(date_sorted_keys)
    
    with open(EXPORT_FILE_NAME, 'w') as finished_file:
        for key in date_sorted_keys:
            finished_file.write(f'{key[0]}\n')
    
    
    return EXPORT_FILE_NAME


if __name__ == "__main__":
    create_aws_clean_list()