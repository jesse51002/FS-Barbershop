import cv2
import os


INPUT_FOLDER = "./custom_images"
OUTPUT_FOLDER = "./input/face"

accepted_format = ["jpeg", "jpg", "pmg"]

output_size = 1024

for img in os.listdir(INPUT_FOLDER):
    basename = img.split(".")[0]
    if img.split(".")[-1] not in accepted_format:
        continue

    img = cv2.imread(os.path.join(INPUT_FOLDER, img))
    img = cv2.resize(img, (output_size, output_size))
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, basename + ".png"), img)