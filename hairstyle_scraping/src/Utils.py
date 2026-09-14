import json
import os
import math
import Constants
import re
import numpy as np
import threading
import pandas as pd
try:
    import thread
except ImportError:
    import _thread as thread
import sys

img_types = ['png', 'jpeg', 'jpg', 'webp']

def setStopFile(stop : bool):
    file = open(Constants.STOP_FILE, 'w')
    json.dump({"Stop": stop}, file)
    file.close()
    
def getStop():
    with open(Constants.STOP_FILE, 'r') as file:
        stop = json.load(file)
        return(stop["Stop"])

# Finds all the images in a directory
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

def get_gan_data_priorty_list():
    if not os.path.isfile(Constants.DATAFRAME_SAVE_FILE):
        return find_images(Constants.CLEAN_BODY_IMAGES_DIR)
    
    dataframe = pd.read_csv(Constants.DATAFRAME_SAVE_FILE)
    
    priority_list = []
    for i in range(dataframe.shape[0]):
        priority = 0
        
        if dataframe.loc[i, "HandDown"]:
            priority += 5
        
        if dataframe.loc[i, "HairType"] in ["braids and dreads", "afro hair", "curly hair"]:
            priority += 3 
            
        if dataframe.loc[i, "Sex"] == "Male":
            priority += 1
            
        priority_list.append(priority)
    
    # Inserts then sorts by priority
    dataframe.insert(0, "Priority", priority_list, True)
    dataframe = dataframe.sort_values(by=['Priority'], ascending=False, ignore_index=True)
    
    priority_pth_list = []
    for i in range(dataframe.shape[0]):
        if dataframe.loc[i, "ImagePath"] is None or pd.isna(dataframe.loc[i, "ImagePath"]):
            continue
        
        relevant_pth = "/".join(dataframe.loc[i, "ImagePath"].split("/")[-2:])
        img_pth = os.path.join(Constants.CLEAN_BODY_IMAGES_DIR, relevant_pth)
        if os.path.isfile(img_pth):
            priority_pth_list.append(img_pth)
    
    return priority_pth_list

# Finds all the images in a directory
def delete_empty(path: str):    
    if os.path.isfile(path):
        return
    if os.path.isdir(path):
        pth_list = os.listdir(path)
        if len(pth_list) == 0:
            os.rmdir(path)
            return
        
        for p in pth_list:
            delete_empty(os.path.join(path, p))
        
        if len(os.listdir(path)) == 0:
            os.rmdir(path)

    
    
def split_hair_dict(hair_dict: dict, count : int):
    # Reads file if exists
    already_scraped = []
    if os.path.isfile(Constants.FINIHSED_RAW_TXT):
        with open(Constants.FINIHSED_RAW_TXT, 'r') as scrape_file:
            already_scraped = [x.strip() for x in scrape_file.readlines()]
    
    keys_to_remove = []
    
    # Removes values that have already been scraped
    for style in hair_dict.keys():
        i = 0
        while i < len(hair_dict[style]):
            style_type = hair_dict[style][i]
            together = f'{style}/{style_type}'
                
            if together in already_scraped:
                hair_dict[style].pop(i)
                print(f"Already scraped {together}, skipping...")
            else:
                i += 1
                    
        # Removes style if all the style_types have been completed
        if len(hair_dict[style]) == 0:
            keys_to_remove.append(style)
    
    for key in keys_to_remove:
        del hair_dict[key]
    
    flattened_dict_arr: list[tuple[str, str]] = []
    for key in hair_dict.keys():
        for value in hair_dict[key]:
            flattened_dict_arr.append((key, value))
  
    split_dicts = [{} for _ in range(count)]
    
    # calcualtes the amount of keys per dictionary
    amount_per = math.ceil(len(flattened_dict_arr) / count)
    
    # splits the dictionary
    global_i = 0
    for i in range(count):
        cur_flattened_arr = []
        for _ in range(amount_per):
            if global_i >= len(flattened_dict_arr):
                break
        
            cur_flattened_arr.append(flattened_dict_arr[global_i])
            global_i += 1
        
        for value in cur_flattened_arr:
            if value[0] in split_dicts[i]:
                split_dicts[i][value[0]].append(value[1])
            else:
                split_dicts[i][value[0]] = [value[1]]
        
    return split_dicts


def split_body_arr(arr, count : int):
    # Reads file if exists
    already_scraped = []
    if os.path.isfile(Constants.FINIHSED_BODY_RAW_TXT):
        with open(Constants.FINIHSED_BODY_RAW_TXT, 'r') as scrape_file:
            already_scraped = [x.strip() for x in scrape_file.readlines()]
    
    for x in already_scraped:
        if x in arr:
            arr.remove(x)
        
    amount_per = math.ceil(len(arr) / count)
    split_arrs = [[] for _ in range(count)]
    
    # splits the dictionary
    global_i = 0
    for i in range(count):
        for j in range(amount_per):
            if global_i >= len(arr):
                break
            split_arrs[i].append(arr[global_i])
            global_i += 1
    
    return split_arrs

def split_group_arr(arr, count : int):
    # Reads file if exists
    already_scraped = []
    if os.path.isfile(Constants.FINIHSED_BODY_RAW_TXT):
        with open(Constants.FINIHSED_BODY_RAW_TXT, 'r') as scrape_file:
            already_scraped = [x.strip() for x in scrape_file.readlines()]
    
    for x in already_scraped:
        for group in arr:
            if group[0] == x:
                arr.remove(group)
                break
        
    amount_per = math.ceil(len(arr) / count)
    split_arrs = [[] for _ in range(count)]
    
    # splits the dictionary
    global_i = 0
    for i in range(count):
        for j in range(amount_per):
            if global_i >= len(arr):
                break
            split_arrs[i].append(arr[global_i])
            global_i += 1
    
    return split_arrs

def get_file_path(pth : str, root_dir : str):
    clean_idx = pth.index(root_dir)
    pth_start = clean_idx + len(root_dir) + 1
    pth_list = "/".join(re.split('[/\\\\]', pth[pth_start:])[:-1])
    return pth_list


def get_flickr_creds():
    # Loads Styles lines from txt file
    # Ignoring comments
    
    assert(os.path.isfile(Constants.FLICKR_CREDS_FILE)), "Flickr creds file doesnt exist"
        
    with open(Constants.FLICKR_CREDS_FILE,'r') as file:
        lines = [x.strip() for x in file.readlines()]
    
    assert(len(lines) >= 2), "Flickr creds file doesnt contain either the key or secret"
    
    key = lines[0]
    secret = lines[1]
    
    return key, secret

def exit_after(s):
    '''
    use as decorator to exit process if 
    function takes longer than s seconds
    '''
    
    def quit_function(fn_name):
        # print to stderr, unbuffered in Python 2.
        print('{0} took too long'.format(fn_name), file=sys.stderr)
        sys.stderr.flush() # Python 3 stderr is likely buffered.
        thread.interrupt_main() # raises KeyboardInterrupt
        
    
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


def vis_seg(pred):
    num_labels = 19

    color = np.array([[0, 0, 0],  ## 0
                      [102, 204, 255],  ## 1
                      [255, 204, 255],  ## 2
                      [255, 255, 153],  ## 3
                      [255, 255, 153],  ## 4
                      [255, 255, 102],  ## 5
                      [51, 255, 51],  ## 6
                      [0, 153, 255],  ## 7
                      [0, 255, 255],  ## 8
                      [0, 0, 255],  ## 9
                      [204, 102, 255],  ## 10
                      [0, 153, 255],  ## 11
                      [0, 255, 153],  ## 12
                      [0, 51, 0],  # 13
                      [102, 153, 255],  ## 14
                      [255, 153, 102],  ## 15
                      [255, 255, 0],  ## 16
                      [255, 0, 255],  ## 17
                      [255, 255, 255],  ## 18
                      ])
    h, w = np.shape(pred)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     print(color.shape)
    for ii in range(num_labels):
        #         print(ii)
        mask = pred == ii
        rgb[mask, None] = color[ii, :]
    # Correct unk
    unk = pred == 255
    rgb[unk, None] = color[0, :]
    return rgb