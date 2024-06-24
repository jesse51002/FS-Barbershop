import os
import cv2
import numpy as np

def calculate_norm_values(root):
    
    total_sum = 0
    total_count = 0
    
    for texture in os.listdir(root):
        texture_folder = os.path.join(root, texture)

        for img in os.listdir(texture_folder):
            img_path = os.path.join(texture_folder, img)
            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            img_avg = img.mean()
            total_sum += img_avg
            total_count += 1
            
    average = total_sum / total_count

    print("Average: ", average)
    
    std_sum = 0
    std_count = 0
    
    for texture in os.listdir(root):
        texture_folder = os.path.join(root, texture)

        for img in os.listdir(texture_folder):
            img_path = os.path.join(texture_folder, img)
            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            relative_img =  np.square(img - average)
            img_sum = relative_img.sum()
            img_count = relative_img.shape[1] * relative_img.shape[0]
            
            std_sum += img_sum
            std_count += img_count
            
    std = np.sqrt(std_sum / std_count)
    
    print("Standard Deviation: ", std)
            
    
if __name__ == "__main__":
    calculate_norm_values("./data/hair_only")
