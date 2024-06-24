import os
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2

from models.FaceDetector import output_bb
from models.FacerParsing import FacerModel, FacerDetection, FACER_CLASSES
from models.p3m_matting.inference import remove_background
from models.face_parsing.inference import get_segmentation, IBUG_CLASSES


data_folder = "./data"
hair_only_folder = os.path.join(data_folder, "hair_only")
full_img_folder = os.path.join(data_folder, "full_img")

BATCH_SIZE = 3
BACKGROUND_COLOR = 255


@torch.no_grad()
def preprocess_dataset():
    """
    Preprocesses the dataset by generating hair-only images for each hair type.
    """
    # Initialize the facer model with a face detector
    facer = FacerModel(face_detector=FacerDetection(), device="cuda")
    
    # Iterate over each hair type
    for hair_type in os.listdir(full_img_folder):
        print("Starting", hair_type)
        
        hair_type_path = os.path.join(full_img_folder, hair_type)
        cur_hair_only_folder = os.path.join(hair_only_folder, hair_type)

        os.makedirs(cur_hair_only_folder, exist_ok=True)
        
        # Get a list of image paths for the current hair type
        imgs = [os.path.join(hair_type_path, img) for img in os.listdir(hair_type_path)]
        done_names = set([os.path.splitext(img)[0] for img in os.listdir(cur_hair_only_folder)])

        i = 0
        while i < len(imgs):
            name = os.path.splitext(os.path.basename(imgs[i]))[0]
            if name in done_names:
                imgs.pop(i)
                i -= 1
            i += 1
        
        # Process sub-batches of images
        for i in range(0, len(imgs), BATCH_SIZE):
            print(f"{hair_type}: Model processing: {i} / {len(imgs)}")
            
            end_i = min(i+BATCH_SIZE, len(imgs))
            size = end_i - i

            cur_imgs = imgs[i: end_i]
            
            cur_batched_input = torch.zeros((size, 3, 1024, 1024))
            for j, img_path in enumerate(cur_imgs):
                image = Image.open(img_path)
                np_image = np.array(image.convert('RGB'))
                torch_img = torch.tensor(np_image).permute(2, 0, 1) / 255
                torch_img = F.interpolate(torch_img.unsqueeze(0), size=(1024, 1024), mode='nearest').squeeze()
                cur_batched_input[j] = torch_img

            cur_batched_input = cur_batched_input.cuda()

            print(f"Processing: {cur_imgs}")
                
            bbs = output_bb(cur_batched_input)

            assert bbs.shape[0] == min(i+BATCH_SIZE, len(imgs)) - i, "BBox mismatch"
            
            # Get the human matting segmentation mask for the current batch
            _, human_segs = remove_background(cur_batched_input)
            
            # Get the keypoints results for the current batch
            ibug_sets = get_segmentation(cur_batched_input, bbs).argmax(1).detach().cpu()

            try:
                # Get the face parsing results for the current batch
                seg_targets, _ = facer.inference(cur_batched_input * 255)
                seg_targets = seg_targets.detach().cpu()
            except Exception:
                print(f"Facer failed")
                seg_targets = torch.zeros((bbs.shape[0], 1024, 1024))
            
            # Convert tensors to numpy arrays
            seg_targets = seg_targets.float().numpy()
            human_segs = human_segs.numpy()
            ibug_sets = ibug_sets.numpy()
            
            # Generate a boolean array indicating which pixels are hair
            hair_bool_array = np.where(
                ((seg_targets == FACER_CLASSES["hair"]) |
                    (ibug_sets == IBUG_CLASSES["hair"])) &
                (human_segs > 0.),
                1, 0
                )
            
            # Generate hair-only images
            cur_batched_input = cur_batched_input.detach().cpu().numpy()
    
            # Transpose the input tensor
            cur_batched_input = np.transpose(cur_batched_input, (0, 2, 3, 1)) * 255
    
            hair_only_ouputs = np.where(np.stack([hair_bool_array] * 3, axis=3) == 1, cur_batched_input, BACKGROUND_COLOR)
            hair_only_ouputs = hair_only_ouputs.astype(np.uint8)
            
            # Save each hair-only image
            for j in range(hair_only_ouputs.shape[0]):
                file_name = os.path.splitext(os.path.basename(cur_imgs[j]))[0] + ".png"
                
                target_pth = os.path.join(hair_only_folder, hair_type, file_name)
    
                left, right, bottom, top = calculate_hair_box(hair_bool_array[j])
    
                original_bounded_hair = hair_only_ouputs[j, bottom:top, left:right]
                
                # Turns the bounds into a box
                vertical_size = top - bottom
                horizontal_size = right - left
    
                l_bounds, r_bounds, b_bounds, t_bounds = 0, 0, 0, 0
                
                if vertical_size > horizontal_size:
                    diff = vertical_size - horizontal_size
                    l_bounds = math.floor(diff / 2)
                    r_bounds = math.ceil(diff / 2)
                elif vertical_size < horizontal_size:
                    diff = horizontal_size - vertical_size
                    b_bounds = math.floor(diff / 2)
                    t_bounds = math.ceil(diff / 2)
    
                # Adds padding to sqaure the image
                square_hair_img = cv2.copyMakeBorder(
                    original_bounded_hair,
                    b_bounds,  # bottom
                    t_bounds,  # top
                    l_bounds,  # left
                    r_bounds,  # right
                    cv2.BORDER_CONSTANT,  # borderType
                    value=[BACKGROUND_COLOR] * 3
                )

                try:
                    img = Image.fromarray(square_hair_img)
                    img = img.convert("L") # Convert to grayscale
                    img.save(target_pth)
                except Exception as e:
                    print(e)


def calculate_hair_box(mask):
    top, bottom, right, left = 0, 0, 0, 0
        
    # Gets positions where the data is not 0
    contains_pos = np.argwhere(mask == 1)
        
    # Handles empty case
    if contains_pos.shape[0] == 0:
        return right, left, bottom, top
        
    # min index where element is not zero
    mins = np.min(contains_pos, axis=0)
    
    bottom = mins[0]
    left = mins[1]
        
    # Max index where element is not zero
    maxs = np.max(contains_pos, axis=0)
    top = maxs[0]
    right = maxs[1]
        
    return left, right, bottom, top


if __name__ == "__main__":
    preprocess_dataset()
            