import os
import sys
sys.path.insert(0,'./')
sys.path.insert(0,'./farl_segmentation')
import time

from ibug.face_parsing import FaceParser as RTNetPredictor

# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import cv2
from  utils.seg_utils import save_original_mask
import numpy as np
import torch

"""
0 : background
1 : skin (including face and scalp)
2 : left_eyebrow
3 : right_eyebrow
4 : left_eye
5 : right_eye
6 : nose
7 : upper_lip
8 : inner_mouth
9 : lower_lip
10 : hair
11 : left_ear
12 : right_ear
13 : glasses
"""

face_parser = RTNetPredictor(
        device="cuda", ckpt="./farl_segmentation/ibug/face_parsing/rtnet/weights/rtnet50-fcn-14.torch", encoder="rtnet50", decoder="fcn", num_classes=14)

MIN_CONFIDENCE = 0.5
# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
# load model
model = YOLO(model_path, task="detect")

# inference
def output_bb(img, confidence = MIN_CONFIDENCE): 
    output = model(img, verbose=False)[0]

        
    cleaned_boxes = []
    for box in output.boxes.data:
        if box[4] > confidence:
            cleaned_boxes.append([
                int(box[0]), # x1
                int(box[1]), # y1
                int(box[2]), # x2
                int(box[3]), # y2
            ])
    
    
    return np.array(cleaned_boxes)

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
        print("Img shape:",img.shape)
        bb = output_bb(img)
        print("Bounding box shape:", bb.shape)
        if len(bb) == 0:
            print(f"No face found for {img_name}")
            continue
    
        seg = get_segmentation(img, bb).argmax(1).detach().cpu().numpy()[0]
        save_original_mask(img_pth, "./output/masks", seg)


    


    
    


    
    