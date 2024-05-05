import cv2
import numpy as np


file_path = "data3.npy"
test_arr = np.load(file_path)

for i in range(20):
    isolated_idx_arr = np.where(test_arr == i, 1, 0)
    cv2.imwrite(f"test_output/{i}.png", isolated_idx_arr * 255)