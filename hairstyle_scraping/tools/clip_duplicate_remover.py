import sys
sys.path.insert(0,'./src')
import os

import numpy as np
from numpy import dot
from numpy.linalg import norm
import cv2
import json

import Constants
from Clip import EmbeddingClip
from Utils import find_images


CLIP_SIMILARITY_THRESHOLD = 0.85

def remove_duplicates_from_hairstyle():
    chosen = -1
    while chosen < 1 or chosen > 2:
        print("""
        Delete dy uploadedalrea folder from?
            1. Raw
            2. Clean
            """)
        
        chosen = int(input())

        if chosen >= 1 or chosen <= 2:
            print(f"""
            You have picked option {chosen}, are sure this action is irreversible\n
            Type 'confirm' to proceed
            """)

            if input() != "confirm":
                print("'confirm' was typed incorrectly. Restart...")
                chosen = -1
                continue
        else:
            print(f"{chosen} is an invalid choice, pick a valid choice")

    root_dir = None
    
    if chosen == 1:
        root_dir = Constants.RAW_IMAGES_DIR
    elif chosen == 2:
        root_dir = Constants.CLEAN_IMAGES_DIR

    num = 0
    removed = 0
    
    styles_pths : list[list[str]] = []
    
    for general_style_folder in os.listdir(root_dir):
        general_folder_pth = os.path.join(root_dir, general_style_folder)
        
        if not os.path.isdir(general_folder_pth):
            continue
        
        for style_folder in os.listdir(general_folder_pth):
            style_folder_pth = os.path.join(general_folder_pth, style_folder)
            
            files = find_images(style_folder_pth)
            files = sorted(files, key=lambda item: os.path.basename(item))
            
            pth_list_file = [file for file in files]
            styles_pths.append(pth_list_file)
            
    file_pth_order : list[tuple[str, np.ndarray]] = []
        
    model = EmbeddingClip()
        
    no_file_found = False
    current_i = 0
    while not no_file_found:
        no_file_found = True
        for style_i in range(len(styles_pths)):
            if len(styles_pths[style_i]) > current_i:
                cur_pth = styles_pths[style_i][current_i]
                latent = model.inference(cv2.imread(cur_pth)).detach().cpu().numpy().flatten()
                file_pth_order.append((styles_pths[style_i][current_i], latent))
                no_file_found = False
            
                if len(file_pth_order) % 100 == 0:
                    print("Finished clip embedding for", len(file_pth_order), "files")
                
        current_i += 1  
            
    duplicate_list = []
        
    current_i = 0
    while current_i < len(file_pth_order):
        original_path, latent = file_pth_order[current_i]
        
        current_j = current_i + 1
        while current_j < len(file_pth_order):
            compare_pth, compare_latent = file_pth_order[current_j]
            
            cos_sim = dot(latent, compare_latent)/(norm(latent)*norm(compare_latent))
            
            if cos_sim >= CLIP_SIMILARITY_THRESHOLD:
                # os.remove(compare_pth)
                # file_pth_order.pop(current_j)
                print(f"{removed}:", original_path, "Removed", compare_pth, "with cos sim", cos_sim)
                duplicate_list.append([original_path, compare_pth, float(cos_sim)])
                removed += 1
                num += 1
                #  current_j -= 1
                
            
            current_j += 1
            
        num += 1
        current_i += 1
        
        if num % 100 == 0:
            print(f"Went through {num} images and removed {removed} duplicates")
        
    with open(os.path.join(root_dir, "duplicate_list.json"), "w") as f:
        json.dump(
            {"duplicates": duplicate_list},
            f
        )
            
                
    print(f"Went through {num} images and removed {removed} duplicates")


if __name__ == "__main__":
    chosen = -1
    while chosen < 1 or chosen > 2:
        print("""
        Delete duplicate images from?
        1. Gan Dataset
        2. Hairstyle presets
        """)
        chosen = int(input())
        

        if chosen < 1 or chosen > 2:
            print(f"{chosen} is an invalid choice, pick a valid choice")
            
    if chosen == 1:
        raise NotImplementedError("Gan Dataset not implemented yet")
    elif chosen == 2:
        remove_duplicates_from_hairstyle()
