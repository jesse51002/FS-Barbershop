import os
import sys
sys.path.insert(0, './')

os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"
import boto3

import json
from presets.Constants import PRESET_PROCESSING_VERSION

PRESET_INDEX = "{:08f}"
PRESET_VERSION_FORMAT = "{:03f}"
PRESET_PROCESSING_FORMAT = "{:03f}"

FULL_ID_FORMAT = f"{PRESET_INDEX}.{PRESET_PROCESSING_FORMAT}"


s3resource = boto3.client('s3')
BUCKET_NAME = "fusion-styles-data"
S3_ID_JSON_KEY = "hairstyle_presets/presets_ids.json"


PRESET_VERSION_V0 = 0
ID_TO_NAME_V0 = {
    PRESET_INDEX.format(0): {
        "name": "shoulder length wavy wispy bangs",
        "raw_img_id": "",
        "version": PRESET_VERSION_FORMAT.format(0)
    },
    PRESET_INDEX.format(1): {
        "name": "shoulder_length_wavy_curtain_bangs.jpeg",
        "raw_img_id": "",
        "version": PRESET_VERSION_FORMAT.format(0)
    },
    PRESET_INDEX.format(2): {
        "name": "long wavy curtain bangs",
        "raw_img_id": "",
        "version": PRESET_VERSION_FORMAT.format(0)
    },
    PRESET_INDEX.format(3): {
        "name": "short curly side swept bangs",
        "raw_img_id": "",
        "version": PRESET_VERSION_FORMAT.format(0)
    },
}
