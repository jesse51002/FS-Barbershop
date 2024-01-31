import os
import sys
sys.path.insert(0,'./')

import cv2
import math
from multiprocessing import Process
import time
import matplotlib.pyplot as plt
import numpy as np
from farl_segmentation.seg_export import get_segmentation, output_bb
from utils.seg_utils import save_original_mask
import torch

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

MUTLIPROCESS = False
PROCESS_COUNT = 3

SIZE_FACE_MULT = 3.5
ANGLE_FACE_MULT = 2
MINIMUM_QUALITY = 256



MASK_FOLDER  = "./DiffusionDatacreator/humanparsing/SegmentationClassAug"
IMAGES_FOLDER  = "./DiffusionDatacreator/humanparsing/JPEGImages"

PARSED_ROOT = "./DiffusionDatacreator/ParsedData"
PARSED_IMAGES = os.path.join(PARSED_ROOT, "images")
PARSED_MASKS = os.path.join(PARSED_ROOT, "masks")
PARSED_MASKS_VIS = os.path.join(PARSED_ROOT, "masks_vis")

body_parse_classes = {
    "background":   0,
    "hat":         1,
    "hair":           2,
    "sunglass":       3,
    "upper-clothes":   4,
    "skirt":          5,
    "pants":           6,
    "dress":           7,
    "belt":            8,
    "left-shoe":       9,
    "right-shoe":      10,
    "face":            11,
    "left-leg":        12,
    "right-leg":       13,
    "left-arm":        14,
    "right-arm":       15,
    "bag":             16,
    "scarf":           17
}

""" FACE PARSE CLASSES
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


def crop_image(img, mask, res_check=False, bottom_extend=False, max_faces=1):
    # Detect faces
    faces = output_bb(img, confidence = 0.75)

    # No face detected
    if len(faces) == 0:
        print("No faces detected... Rejected")
        return None, None, None
    
    if max_faces is not None and len(faces) > max_faces:
        print("To many faces detected... Rejected")
        return None, None, None
    
    x, y, w, h = faces[0][0], faces[0][1], faces[0][2] - faces[0][0], faces[0][3] - faces[0][1]

    # img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

    
    # return img, img[:,:,0], (x, y, w, h)
    mid_x, mid_y = int(x + w/2), int(y + h/2)
    max_size = int(max(w,h) / 2)
        
    total_res = int(max_size * SIZE_FACE_MULT * 2)
        
    # Resolution checker
    if res_check and total_res < MINIMUM_QUALITY:
        print(f"{total_res} Resolution wasnt big enough... Rejected")
        return None, None, None
        
    bounds =  (
        int(mid_x - max_size * SIZE_FACE_MULT), # Bottom
        int(mid_x + max_size * SIZE_FACE_MULT), # top
        int(mid_y - max_size * SIZE_FACE_MULT), # left
        int(mid_y + max_size * SIZE_FACE_MULT) # right
    )
        
        
    b,t,l,r = bounds 
        
    b_bounds, t_bounds, l_bounds, r_bounds = (
        max(0,-1*b),
        max(0,t - img.shape[1]),
        max(0,-1*l), 
        max(0,r - img.shape[0])
        )
        
    if r_bounds > 0:
        print("Bottom doesnt reach... Rejected")
        return None, None, None
            
    bounded_image = img[
        max(0, l): min(img.shape[0], r),
        max(0, b): min(img.shape[1], t)
        ]
        
    bounded_image = cv2.copyMakeBorder(
        bounded_image, 
        l_bounds,  # left
        r_bounds, # right
        b_bounds, # bottom
        t_bounds, #top
        cv2.BORDER_CONSTANT #borderType
        )

    bounded_mask = mask[
        max(0, l): min(mask.shape[0], r),
        max(0, b): min(mask.shape[1], t)
        ]

    bounded_mask = cv2.copyMakeBorder(
        bounded_mask, 
        l_bounds,  # left
        r_bounds, # right
        b_bounds, # bottom
        t_bounds, #top
        cv2.BORDER_CONSTANT #borderType
        )

    center = int(max_size * SIZE_FACE_MULT)
            
    return bounded_image, bounded_mask, (int(center - w / 2), int(center - h / 2), int(center + w / 2), int(center + h / 2))


def create_mask(img_pth, mask_pth):
    img = cv2.imread(img_pth)
    mask = cv2.imread(mask_pth)[:, :, 0]

    hat = np.where(mask == body_parse_classes["hat"])
    if len(hat[0]) > 0:
        print("Mask contains hat, skipping...")
        return None, None, None

    # Mask touching edges check
    test_areaY = np.stack([mask[:, :5], mask[:, -5:]], axis=1)
    test_areaX= mask[:5]
        
    invlaid_idxsY = np.where(test_areaY != 0)
    invlaid_idxsX = np.where(test_areaX != 0)
        
    if invlaid_idxsY[0].shape[0] > 0 or invlaid_idxsX[0].shape[0] > 0:
        print("Body is not contained in the image... Rejected")
        return None, None, None 
    
    bounded_image, bounded_mask, bounding_box = crop_image(img, mask)

    if bounded_image is None:
        return None, None, None

    down_face_seg = get_segmentation(bounded_image, [bounding_box])
    face_seg = torch.argmax(down_face_seg, dim=1).long().cpu().numpy()[0]

    result_mask = np.zeros_like(bounded_mask)

    # Adds hair
    result_mask[bounded_mask == body_parse_classes["hair"]] = 10

    # Adds the rest body from the body parse mask
    result_mask[(bounded_mask != body_parse_classes["face"]) & (bounded_mask != body_parse_classes["background"])] = 14
    # Adds face segmentation features
    # Doesnt add hair since we already have hair groundtruth
    result_mask = np.where(
        (
             (result_mask == 0) & ((bounded_mask == body_parse_classes["face"]) | (bounded_mask == body_parse_classes["sunglass"]) )
        ), 
        face_seg, 
        result_mask
    )
    
    # Only keeps the body mask below the nose
    # The face segmentation model will handle the face area
    nose_idx =  np.where(face_seg == 6)
    if len(nose_idx[0]) == 0:
        print("No nose found")
        return None, None, None
    max_nose = nose_idx[0].max()
        
    bounded_mask_below_nose = bounded_mask.copy()
    bounded_mask_below_nose[:max_nose] = 0

    # Adds body below face to the mask
    result_mask = np.where((bounded_mask_below_nose != 0) & (result_mask == 0), 14, result_mask)

    # Fills in face areas that might've been missed
    bounded_mask_above_nose = bounded_mask.copy()
    bounded_mask_above_nose[max_nose:] = 0
    result_mask = np.where((bounded_mask_above_nose != 0) & (result_mask == 0), 1, result_mask)
    
    
    return bounded_image.astype(np.uint8), bounded_mask.astype(np.uint8),result_mask.astype(np.uint8)

@torch.no_grad()
def parse_names(names):
    count = 0
    failed_count = 0

    scale_target = 256
    for name in names:
        print(name)
        
        img_pth = os.path.join(IMAGES_FOLDER, name)
        
        mask_file_name = name.split(".")[0] + ".png"

        mask_pth = os.path.join(MASK_FOLDER, mask_file_name)

        bounded_image, bounded_mask, result_mask = create_mask(img_pth, mask_pth)

        if bounded_image is None:
            print("Faied to make image")
            failed_count += 1
            continue
        
        bounded_image = cv2.resize(bounded_image, (scale_target,scale_target))
        result_mask = cv2.resize(result_mask, (scale_target,scale_target), interpolation=cv2.INTER_NEAREST)

        output_img_pth = os.path.join(PARSED_IMAGES, name)
        output_mask_pth = os.path.join(PARSED_MASKS, mask_file_name)
        
        cv2.imwrite(output_img_pth, bounded_image)
        cv2.imwrite(output_mask_pth, result_mask)
        
        display_new_mask = save_original_mask(mask_file_name, PARSED_MASKS_VIS, result_mask, return_numpy=True)
        body_mask = save_original_mask(mask_file_name, PARSED_MASKS_VIS, bounded_mask, return_numpy=True)

        # setting values to rows and column variables 
        rows = 1
        columns = 3
        
        # create figure 
        fig = plt.figure(figsize=(10, 7)) 

        # Image
        fig.add_subplot(rows, columns, 1) 
        plt.imshow(bounded_image[:,:,::-1]) 
        plt.axis('off') 
        plt.title("Image") 
        
        # Generated Mask
        fig.add_subplot(rows, columns, 2) 
        plt.imshow(display_new_mask)
        plt.axis('off') 
        plt.title("Generated mask") 

        # Og body mask
        fig.add_subplot(rows, columns, 3) 
        plt.imshow(body_mask)
        plt.axis('off') 
        plt.title("Body mask")

        
        plt.savefig(os.path.join(PARSED_MASKS_VIS, mask_file_name))
        
        print(f"Finished creating data: {count}")
        count += 1

    print(f"Created {count} images and failed {failed_count}")
          

def create_dataset():
    if not os.path.isdir(PARSED_IMAGES):
        os.makedirs(PARSED_IMAGES)
    if not os.path.isdir(PARSED_MASKS):
        os.makedirs(PARSED_MASKS)
    if not os.path.isdir(PARSED_MASKS_VIS):
        os.makedirs(PARSED_MASKS_VIS)

    names = os.listdir(IMAGES_FOLDER)

    done_names = os.listdir(PARSED_IMAGES)

    for done in done_names:
        if done in names:
            names.remove(done)

    if MUTLIPROCESS :
        num_per = math.ceil(len(names) / PROCESS_COUNT)
    
        name_splits = [[] for _ in range(PROCESS_COUNT)]
    
        global_i = 0
        for i in range(PROCESS_COUNT):
            for _ in range(num_per):
                if global_i >= len(names):
                    break
                name_splits[i].append(names[global_i])
                global_i += 1
    
        torch.multiprocessing.set_start_method('spawn')
        
        processes = []
        for i in range(PROCESS_COUNT):
            p = Process(target=parse_names, args=(name_splits[i],))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    else:
        parse_names(names)
        

        
if __name__ == "__main__":
    create_dataset()

    

    