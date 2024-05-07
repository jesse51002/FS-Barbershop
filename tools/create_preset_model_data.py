import os
import sys
sys.path.insert(0, './')

os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"
import boto3

import time

from args_maker import create_parser
from tools.upload_raw_image_presets import S3_RAW_IMG_PRESETS_KEY

from models.FacerParsing import FacerModel, FacerKeypoints, FacerDetection
from models.p3m_matting.inference import human_matt_model
from models.Preprocessor import Preprocessor
from models.SegmentationMaker import SegMaker
from models.Embedding import Embedding
from models.Net import Net

S3_MODEL_DATA_PRESETS_KEY = "hairstyle_presets/presests_model_outputs/"
s3resource = boto3.client('s3')
BUCKET_NAME = "fusion-styles-data"

REQUIRED_DATA = {
    "cleaned_img": "{0}.png",
    "w+": "{0}_w+.npy",
    "fs": "{0}_fs.npz",
    "mask": "{0}_mask.npz"
}


def get_uploaded_raw_keys():
    response = s3resource.list_objects(
        Bucket=BUCKET_NAME,
        Prefix=S3_RAW_IMG_PRESETS_KEY,
        MaxKeys=10000  # s3 list_object limit will probably be reached before this
    )

    # Parses the response to get the names
    keys = list(set([x["Key"] for x in response["Contents"]]))
    # Remove empty names (these come from folder keys)
    if S3_RAW_IMG_PRESETS_KEY in keys:
        keys.remove(S3_RAW_IMG_PRESETS_KEY)

    return keys


def download_raw_images(opts):
    uploaded_raw_images = get_uploaded_raw_keys()
    
    img_names = [os.path.basename(x).split(".")[0] for x in uploaded_raw_images]

    images_to_download = []
    
    for i in range(len(uploaded_raw_images)):
        # Checks to make sure if all the needed data is already on s3
        # if the data is already on s3 then skip
        
        preset_model_data_key = S3_MODEL_DATA_PRESETS_KEY + img_names[i] + "/"
        
        response = s3resource.list_objects(
            Bucket=BUCKET_NAME,
            Prefix=preset_model_data_key,
            MaxKeys=10000  # s3 list_object limit will probably be reached before this
        )

        # If the folder doesnt exist yet
        if "Contents" not in response:
            images_to_download.append(uploaded_raw_images[i])
            continue
            
        # Parses the response to get the names
        names = list(set([os.path.basename(x["Key"]) for x in response["Contents"]]))
        # Remove empty names (these come from folder keys)
        if '' in names:
            names.remove('')

        # Gets all the need files in the s3 and makes sure they are in the bucket
        # if they are not then we download it for data creation and uploading
        img_needed_files = [x.format(img_names[i]) for x in REQUIRED_DATA.values()]

        not_done_files = set(img_needed_files) - set(names)
        if len(not_done_files) > 0:
            images_to_download.append(uploaded_raw_images[i])
        else:
            print(img_names[i], "already has the needed model data..... skipping")

    local_raw_img_pths = [os.path.join(opts.unprocessed, os.path.basename(x)) for x in images_to_download]

    if not os.path.isdir(opts.unprocessed):
        os.makedirs(opts.unprocessed)
    
    for i in range(len(images_to_download)):
        print("Downloading to:", local_raw_img_pths[i])
        s3resource.download_file(Bucket=BUCKET_NAME, Key=images_to_download[i], Filename=local_raw_img_pths[i])

    return local_raw_img_pths


def upload_model_data_presets(opts, names):
    for name in names:
        print("Uploading model data for:", name)
        data_folder_key = S3_MODEL_DATA_PRESETS_KEY + name + "/"
        
        local_to_keys = [
            [
                data_folder_key + REQUIRED_DATA["cleaned_img"].format(name),
                os.path.join(opts.input_dir, REQUIRED_DATA["cleaned_img"].format(name))
            ],
            [
                data_folder_key + REQUIRED_DATA["w+"].format(name),
                os.path.join(opts.output_dir, "W+", name + ".npy")
            ],
            [
                data_folder_key + REQUIRED_DATA["fs"].format(name),
                os.path.join(opts.output_dir, "FS", name + ".npz")
            ],
            [
                data_folder_key + REQUIRED_DATA["mask"].format(name),
                os.path.join(opts.output_dir, "masks", REQUIRED_DATA["mask"].format(name))
            ],
        ]

        # Uploads the model preset data
        for loc_to_key in local_to_keys:
            s3resource.upload_file(loc_to_key[1], BUCKET_NAME, loc_to_key[0])
            
        print("Finished uploading model data for:", name)


def create_preset_model_data():
    grand_start_time = time.time()

    raw_args = [
        "--W_steps", "1000",
        "--FS_steps", "250",
        "--learning_rate", "0.01",
        "--sign", "realistic",
        "--smooth", "5",
        "--unprocessed", "input/preset_unprocessed",
        "--input_dir", "input/preset_face",
        "--output_dir", "output/preset_output"
    ]
    
    parser = create_parser()
    args = parser.parse_args(raw_args)
    
    raw_imgs_pth = download_raw_images(args)

    if len(raw_imgs_pth) == 0:
        print("Nothing new to create model data for!")
        return
    
    net = Net(args)
    face_detector = FacerDetection()
    keypoint_model = FacerKeypoints(face_detector=face_detector, device=args.device)
    facer = FacerModel(face_detector=face_detector, device=args.device)
    background_remover = human_matt_model(device=args.device)

    print("Starting preprocessor")
    start = time.time()
    preprocessor = Preprocessor(args, keypoint_model=keypoint_model, background_remover=background_remover)
    preprocessor.preprocess_imgs(raw_imgs_pth)
    
    print(f"preprocessor {time.time() - start}")
    
    print("Starting segmentor")
    start = time.time()
    segmentor = SegMaker(args, facer=facer, background_remover=background_remover, keypoint_model=keypoint_model)

    clean_img_paths = [os.path.join(args.input_dir, os.path.basename(x).split(".")[0] + ".png") for x in raw_imgs_pth]
    
    segmentor.create_segmentations(clean_img_paths)
    print(f"segmentor {time.time() - start}")
    
    print("Starting ai space creation")
    start = time.time()
    ii2s = Embedding(args, net=net)
    
    print("Invert in W")
    ii2s.invert_images_in_W(clean_img_paths)
    print("Invert in FS")
    ii2s.invert_images_in_FS(clean_img_paths)
    print(f"Embedding took {time.time() - start}")

    print("Uploading preset data")
    names = [os.path.basename(x).split(".")[0] for x in raw_imgs_pth]
    upload_model_data_presets(args, names)
    
    print(f"Total preset making time {(time.time() - grand_start_time) / 60} minutes")

    


if __name__ == "__main__":
    create_preset_model_data()