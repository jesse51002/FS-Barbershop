import os
from enum import Enum
import json

import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
from torchvision import transforms
from torchvision.io.image import ImageReadMode

from matplotlib import pyplot as plt

from tools.train_test_split import SPLIT_JSON

IMG_SIZE = 224
MEAN = 0.72143692445
STD = 0.36642099773



class Split(Enum):
    ALL = 0
    TRAIN = 1
    TEST = 2


class CustomColorJitterTransform(torch.nn.Module):
    
    def __init__(self, brightness=0.5):
        super().__init__()
        self.jitter = transforms.ColorJitter(brightness=brightness)
    
    def forward(self, img):
        """
        Apply color jitter to the image.
        Makes sure that the background stays white 

        Args:
            img (torch.Tensor): The input image with shape (C, H, W) in the range [0, 255].

        Returns:
            torch.Tensor: The transformed image with shape (C, H, W) in the range [0, 255].
        """
        # Saves where the values were originally 255.
        is_back = torch.where(img[0] == 255, 1, 0)

        # Apply color jitter to the image.
        img = self.jitter(img)

        # Restore the pixel values back to 255 if they were originally 255.
        img = torch.where(is_back == 1, 255, img)

        return img


TRANSFORM = v2.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    CustomColorJitterTransform(brightness=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[MEAN], std=[STD]),
])

TRANSFORM_NO_CHANGE = v2.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[MEAN], std=[STD]),
])

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=False),
    v2.Normalize(mean=[MEAN], std=[STD]),
])

INV_TRANSFORM = transforms.Compose([
    transforms.Normalize(
        mean = [ 0.],
        std = [1/STD]),
    transforms.Normalize(
        mean = [-MEAN],
        std = [ 1. ]),
])


class HairTypeDataset(Dataset):
    def __init__(self, img_dir, split=Split.ALL):
        
        self.transform = TRANSFORM
            
        self.class_to_idx = {
            "afro": 0,
            "braids_dreads": 1,
            "curly": 2,
            "men": 3,
            "staright": 4,
            "wavy": 5
        }
        
        self.img_data : list[tuple[str, int]] = []
        self.img_dir = img_dir
        self.split = split
        
        with open(SPLIT_JSON, "r") as f:
            split_info = json.load(f)
        
        valid_list = None
        if split == Split.TEST:
            valid_list = split_info["test"]
        elif split == Split.TRAIN:
            valid_list = split_info["train"]
        
        for folder in os.listdir(self.img_dir):
            folder_pth = os.path.join(self.img_dir, folder)
            
            if not os.path.isdir(folder_pth):
                continue

            for img in os.listdir(folder_pth):
                img_pth = os.path.join(folder_pth, img)
                
                rel_pth = f"{folder}/{img}"
                
                if valid_list is not None and rel_pth not in valid_list:
                    continue
                
                self.img_data.append((img_pth, self.class_to_idx[folder]))

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path = self.img_data[idx][0]
        image = read_image(img_path, mode=ImageReadMode.GRAY)
        label = self.img_data[idx][1]
        if self.transform is not None:
            if self.split == Split.TEST:
                image = TRANSFORM_NO_CHANGE(image)
            else:
                image = self.transform(image)
                
        return image, label
    
    
if __name__ == "__main__":
    dataset = HairTypeDataset("./data/hair_only")

    import math
    import random

    # Make a list of all the indexes
    indexes = list(range(len(dataset)))
    random.shuffle(indexes)

    # Split the list into chunks of 10
    num_batches = len(dataset) // 10
    batches = [indexes[i:i+9] for i in range(0, len(indexes), 9)]

    # Loop through the batches
    for i, batch_indexes in enumerate(batches):
        if i == 1:
            break
        images = []
        for j in batch_indexes:
            image, label = dataset[j]
            image = INV_TRANSFORM(image)
            images.append(image)
        images = torch.stack(images)
        
        # Display the images
        plt.figure(figsize=(10, 10))
        for j in range(images.shape[0]):
            plt.subplot(int(math.sqrt(images.shape[0])), int(math.sqrt(images.shape[0])), j+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(True)
            plt.imshow(images[j, ...].permute(1, 2, 0), cmap='gray')
        plt.show()
