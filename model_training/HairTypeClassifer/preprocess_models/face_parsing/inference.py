import sys
sys.path.insert(0,'./models/face_parsing/')

import os

from ibug.face_parsing import FaceParser as RTNetPredictor
from models.FaceDetector import output_bb
# load libraries

from PIL import Image
import cv2
import numpy as np
import torch


IBUG_CLASSES = {
        'background': 0,
        'face': 1,
        'lbrow': 2,
        'rbrow': 3,
        'leye': 4,
        'reye': 5,
        'nose': 6,
        'ulip': 7,
        'mouth': 8,
        'llip': 9,
        'hair': 10,
        'lear': 11,
        'rear': 12,
        'glasses': 13
    }

face_parser = RTNetPredictor(
        device="cuda", ckpt="./models/face_parsing/ibug/face_parsing/rtnet/weights/rtnet50-fcn-14.torch", encoder="rtnet50", decoder="fcn", num_classes=14)


def get_segmentation(img, bbox):
    masks = face_parser.predict_img(img, bbox, rgb=True)
    return masks


if __name__ == "__main__":
    face_dir = "./input/face/"
    accepted_imgs = ["png", "jpg", "jpeg"]

    torch.no_grad()
    for img_name in os.listdir(face_dir):
        if img_name.split(".")[-1] not in accepted_imgs:
            continue

        print(img_name)
        img_pth = os.path.join(face_dir, img_name)
        unscaled_img = torch.tensor(cv2.imread(img_pth)[:,:,::-1].copy()).permute((2,0,1))
        unscaled_img = unscaled_img.unsqueeze(0).float().to("cuda:0")
        img = unscaled_img / 255
        print("Img shape:", img.shape)
        bb = output_bb(img)
        print("Bounding box shape:", bb.shape)
        if len(bb) == 0:
            print(f"No face found for {img_name}")
            continue

        seg = get_segmentation(img, bb).argmax(1).detach().cpu().numpy()[0]
