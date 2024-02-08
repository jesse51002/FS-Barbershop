import os

import random
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import draw
from mask_viewer import vis_seg


DESTROYED_DATA_ROOT = "./DestroyedData"
GROUND_TRUTH_PTH = os.path.join(DESTROYED_DATA_ROOT, "truth")
VISUALIZE_PTH = os.path.join(DESTROYED_DATA_ROOT, "visualize")

MASK_ROOT = "./ParsedData/masks"
IMAGES_ROOT = "./ParsedData/images"

MAX_CIRCLE_COUNT = 3

DESTROY_TYPES = ["hair_shift", "circle", "crop"]

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

def crop_destroy(mask: np.ndarray, amount=None):    
    crop_dest_mask = mask.copy()
    
    right, left, bottom, top  = calculate_bound_box(crop_dest_mask)
    mid_x, mid_y = int((right + left) / 2), int((bottom + top) / 2)

    if amount is None:
        amount = random.randint(1, len(CROP_DIRECTIONS))
    
    direction_options = CROP_DIRECTIONS.copy()
    
    # Amount that random float is squared to increase the chances of lower numbers
    lower_priority = 4
    
    for _ in range(amount):
        rand_idx = random.randint(0, len(direction_options) - 1)
        
        chosen_dir = direction_options[rand_idx]
        direction_options.pop(rand_idx)
        
        destroy_amount = pow(random.random(), lower_priority)
        
        # print(chosen_dir, destroy_amount)
        
        if chosen_dir == "top":
            start_idx = int(top - destroy_amount * (top - mid_y))
            crop_dest_mask[start_idx:] = 0
        elif chosen_dir == "bottom":
            start_idx = int(bottom + destroy_amount * (top - mid_y))
            crop_dest_mask[:start_idx] = 0
        elif chosen_dir == "left":
            start_idx = int(left - destroy_amount * (left - mid_x))
            crop_dest_mask[:, start_idx:] = 0
        elif chosen_dir == "right":
            start_idx = int(right + destroy_amount * (left - mid_x))
            crop_dest_mask[:, :start_idx] = 0
    
    return crop_dest_mask

def circle_destroy(mask: np.ndarray, amount=None):    
    cir_dest_mask = mask.copy()
    
    right, left, bottom, top  = calculate_bound_box(cir_dest_mask)

    if amount is None:
        amount = random.randint(1, MAX_CIRCLE_COUNT)
        
    min_perc = 0.05
    max_perc = 0.7
    
    # Amount that random float is squared to increase the chances of lower numbers
    lower_priority = 2
    
    
    for _ in range(amount):
        # Calculates the random center
        center_y = int(bottom + (top - bottom) * random.random())
        center_x = int(right + (left - right) * random.random())
        
        # Uses the larger bound direction as a scaling refernce
        square_size = max((top - bottom), (left - right))
        
        # Gets the scaled size of the x and y axis ---
        
        rand_y_max_size = square_size * (min_perc + (max_perc - min_perc) * random.random())
        relative_y_size = min_perc + (max_perc - min_perc) * math.pow(random.random(), lower_priority)
        size_y = int(relative_y_size * (rand_y_max_size * 0.75) + (rand_y_max_size * 0.25))
        
        rand_x_max_size = square_size * (min_perc + (max_perc - min_perc) * random.random())
        relative_x_size = min_perc + (max_perc - min_perc) * math.pow(random.random(), lower_priority)
        size_x = int(relative_x_size * (rand_x_max_size * 0.75) + (rand_x_max_size * 0.25))
        
        # ---
        
        # Gets a ranodm angle
        rand_angle = 2 * math.pi * random.random()
        
        # Calculatse ellipse mask and sets it to zero
        rr, cc = draw.ellipse(center_y, center_x, size_y, size_x, shape=cir_dest_mask.shape, rotation=rand_angle)
        
        cir_dest_mask[rr, cc] = 0
        
    
    return cir_dest_mask

def hair_shift_destroy(mask: np.ndarray, amount=None):
    # All category boundin box
    right, left, bottom, top  = calculate_bound_box(mask)
    
    
    max_move_perc = 0.05
    
    # Amount to move the hair, this works for both positive and negative movement
    hair_move_x = int((left - right) * 2 * random.random() * max_move_perc - ((left - right) * max_move_perc)) 
    hair_move_y = int((top - bottom) * 2 * random.random() * max_move_perc- ((top - bottom) * max_move_perc)) 
    
    
    # hair bounding box
    hair_right, hair_left, hair_bottom, hair_top = calculate_bound_box(np.where(mask == 10, 1, 0))
    
    if hair_right + hair_move_x < 0:
        hair_right += -(hair_right + hair_move_x)
    if hair_bottom + hair_move_y < 0:
        hair_bottom += -(hair_bottom + hair_move_y)
    
    if hair_left + hair_move_x >= mask.shape[1]:
        hair_left -= 1 + (hair_left + hair_move_x) - mask.shape[1]
    if hair_top + hair_move_y >= mask.shape[0]:
        hair_top -= 1 + (hair_top + hair_move_y) - mask.shape[0]
    
    # gets hair mask
    hair_mask = np.where(mask[hair_bottom:hair_top, hair_right:hair_left] == 10, 1, 0)
    
    # Creates a moved mask for the just the hair
    moved_hair_mask_scaled = np.zeros_like(mask)
    moved_hair_mask_scaled[
        hair_bottom + hair_move_y : hair_top + hair_move_y,
        hair_right + hair_move_x : hair_left + hair_move_x
        ] = hair_mask
    
    # Removes current hair to shift it
    mask_hair_removed = mask.copy()
    mask_hair_removed[mask_hair_removed == 10] = 0
    
    # Shifts the current hair
    mask_hair_moved = np.where(moved_hair_mask_scaled == 1, 10, mask_hair_removed)
    
    return mask_hair_moved
        
# Gets all possible destroy combinations in order to make a dataset
def get_combinations(current = [], left_to_add = DESTROY_TYPES):  
    
    if len(left_to_add) == 0:
        if len(current) > 0:
            return [current]
        else:
            return []
    
    reminder_arr = left_to_add[1:]
    
    new_combs = []
    
    new_combs += get_combinations(current + [], reminder_arr)
    new_combs += get_combinations(current + [left_to_add[0]], reminder_arr)
             
    return new_combs
    
def destroy_dataset():
    if not os.path.isdir(VISUALIZE_PTH):
        os.makedirs(VISUALIZE_PTH)
        
    if not os.path.isdir(GROUND_TRUTH_PTH):
        os.makedirs(GROUND_TRUTH_PTH)
    
    
    destroy_combinations = get_combinations()
    
    # Create paths for destroy combinations
    destroy_names = ["_".join(x) for x in destroy_combinations]
    for cur_destroy_name in destroy_names:
        target_dir = os.path.join(DESTROYED_DATA_ROOT, cur_destroy_name)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
    
    
    for mask_name in os.listdir(MASK_ROOT):
        print(mask_name)
        
        mask_pth = os.path.join(MASK_ROOT, mask_name)
        mask = cv2.imread(mask_pth)[:,:,0]
        
        destroyed_masks = []
        
        for combination in destroy_combinations:
            
            current_mask = mask.copy()
            
            # destroys the mask
            for destroy_type in combination:
                if destroy_type == "hair_shift":
                    current_mask = hair_shift_destroy(current_mask)
                elif destroy_type == "circle":
                    current_mask = circle_destroy(current_mask)
                elif destroy_type == "crop":
                    current_mask = crop_destroy(current_mask)
            
            # Appends to destroyed masks list
            destroyed_masks.append(current_mask)
        
        
        # create figure 
        fig = plt.figure(figsize=(10, 7)) 
        
        rows = math.ceil(len(destroyed_masks) / 3)
        columns = 3
        
        # Saves original
        cv2.imwrite(os.path.join(GROUND_TRUTH_PTH, mask_name), mask)
        
        # Visualizes original
        fig.add_subplot(rows, columns, 1) 
        plt.imshow(vis_seg(mask)) 
        plt.axis('off') 
        plt.title("original") 
        
        for i, mask in enumerate(destroyed_masks):
            # Saves the mask
            cv2.imwrite(os.path.join(DESTROYED_DATA_ROOT, destroy_names[i], mask_name), mask)
            
            # Visualzies the destroyed masks
            fig.add_subplot(rows, columns, i + 2) 
            plt.imshow(vis_seg(mask)) 
            plt.axis('off') 
            plt.title(destroy_names[i]) 
        
    
        # plt.show()
        plt.savefig(os.path.join(VISUALIZE_PTH, mask_name))
        plt.close()
        
        
        
if __name__ == "__main__":
    destroy_dataset()