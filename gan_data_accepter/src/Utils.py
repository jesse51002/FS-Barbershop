import json
import os
import math
import re
import numpy as np
import pandas as pd
import sys
import shutil

img_types = ['png', 'jpeg', 'jpg', 'webp']


# Finds all the images in a directory
def delete_empty(path: str):    
    if os.path.isfile(path):
        return
    if os.path.isdir(path):
        if os.path.basename(path) == ".ipynb_checkpoints":
            shutil.rmtree(path)
            return
            
        pth_list = os.listdir(path)
        if len(pth_list) == 0:
            os.rmdir(path)
            return
        
        for p in pth_list:
            delete_empty(os.path.join(path, p))
        
        if len(os.listdir(path)) == 0:
            os.rmdir(path)

def get_file_path(pth : str, root_dir : str):
    clean_idx = pth.index(root_dir)
    pth_start = clean_idx + len(root_dir) + 1
    pth_list = "/".join(re.split('[/\\\\]', pth[pth_start:])[:-1])
    return pth_list


def find_images(path: str):
    img_list = []
    
    if os.path.isdir(path):
        for sub in os.listdir(path):
            if ".ipynb_checkpoints" in sub:
                continue
            img_list += find_images(os.path.join(path, sub))
    else:
        name = os.path.basename(path)
        file_type = name.split(".")[-1]
        if file_type in img_types:
            img_list.append(path)
        
    return img_list