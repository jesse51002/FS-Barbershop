import os

import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

from mask_viewer import vis_seg


MASK_ROOT = "./ParsedData/masks"
IMAGES_ROOT = "./ParsedData/images"

MAX_CIRCLE_COUNT = 3

CIRCLE_DESTROY = 1
CROP_DESTROY = 2

CROP_DIRECTIONS = ["top", "bottom", "right", "left"]

# Calculates the bounding box information from semantic label
def calculate_bound_box(mask):
    top, bottom, right, left = 0,0,0,0 
    
    # Gets positions where the data is not 0
    contains_pos = np.argwhere(mask != 0)
    
    # Handles empty case
    if contains_pos.shape[0] == 0:
        return right, left, bottom, top
    
    # min index where element is not zero
    mins = np.min(contains_pos, axis = 0)
    
    bottom = mins[0]
    right = mins[1]
    
    # Max index where element is not zero
    maxs = np.max(contains_pos, axis = 0)
    top = maxs[0]
    left = maxs[1]
    
    return right, left, bottom, top

def crop_destroy(mask, amount=None):    
    right, left, bottom, top  = calculate_bound_box(mask)
    mid_x, mid_y = int((right + left) / 2), int((bottom + top) / 2)

    if amount is None:
        amount = random.randint(1, len(CROP_DIRECTIONS))
    
    direction_options = CROP_DIRECTIONS.copy()
    
    for _ in range(amount):
        rand_idx = random.randint(0, len(direction_options) - 1)
        
        chosen_dir = direction_options[rand_idx]
        direction_options.pop(rand_idx)
        
        destroy_amount = pow(random.random(), 3)
        
        # print(chosen_dir, destroy_amount)
        
        if chosen_dir == "top":
            start_idx = int(top - destroy_amount * (top - mid_y))
            mask[start_idx:] = 0
        elif chosen_dir == "bottom":
            start_idx = int(bottom + destroy_amount * (top - mid_y))
            mask[:start_idx] = 0
        elif chosen_dir == "left":
            start_idx = int(left - destroy_amount * (left - mid_x))
            mask[:, start_idx:] = 0
        elif chosen_dir == "right":
            start_idx = int(right + destroy_amount * (left - mid_x))
            mask[:, :start_idx] = 0
    
    return mask

def circle_destroy(mask, amount=None):    
    right, left, bottom, top  = calculate_bound_box(mask)

    if amount is None:
        amount = random.randint(1, MAX_CIRCLE_COUNT)
    
    
    
    return mask
        
        
def destroy_dataset():
    
    for mask_name in os.listdir(MASK_ROOT):
        print(mask_name)
        
        mask_pth = os.path.join(MASK_ROOT, mask_name)
        
        mask = cv2.imread(mask_pth)[:,:,0]
        
        cropped_mask = crop_destroy(mask)

        plt.imshow(vis_seg(cropped_mask))
        plt.show()
        
        
        
if __name__ == "__main__":
    destroy_dataset()