import os
import cv2
import numpy as np
import random
import json

SPLIT_JSON = "train_test_split.json"
TRAIN_PERCENTAGE = 0.8
SEED = 9801723987

def train_test_split(root):
    random.seed(SEED)
    train_test_split = {
        "train": [],
        "test": []
    }
    
    for texture in os.listdir(root):
        texture_folder = os.path.join(root, texture)

        imgs_pth = [f"{texture}/{img}" for img in os.listdir(texture_folder)]
        
        random.shuffle(imgs_pth)
        
        train_end_idx = int(len(imgs_pth) * TRAIN_PERCENTAGE)
        
        train_list = imgs_pth[:train_end_idx]
        test_list = imgs_pth[train_end_idx:]
        
        train_test_split["train"].extend(train_list)
        train_test_split["test"].extend(test_list)
        
    print("Train: {}, Test: {}".format(len(train_test_split["train"]), len(train_test_split["test"])))
        
    print("Saving train test split to {}".format(SPLIT_JSON))
    with open(SPLIT_JSON, "w") as f:
        json.dump(train_test_split, f)
        
            
if __name__ == "__main__":
    train_test_split("./data/hair_only")
