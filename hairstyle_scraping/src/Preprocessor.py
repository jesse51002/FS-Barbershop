import sys
sys.path.insert(0, './tools')

import os
import cv2
import time
import math
import torch
from Utils import getStop, get_file_path, find_images
from sixdrepnet import SixDRepNet
import matplotlib.pyplot as plt
import numpy as np

from HumanParsing.segment import body_model
from Detection import detection_model
from Clip import Clip, HairClip
import Constants

from aws_s3_downloader import download_aws_folder, get_download_folders
from threading import Thread, Lock

SIZE_FACE_MULT = 3.5
ANGLE_FACE_MULT = 2
MINIMUM_QUALITY = 1024

MAX_QUALITY = 1024

# This is mirrored to negative angles as well
# Format
# [ 
#   pitch (up and down),
#   yaw  (side to side),
#   roll (angling side to side)
# ]

ALLOWED_ANGLES = {
    "front": np.array([[0, 13], [0, 13], [0, 13]]),
    "side": np.array([[0, 35], [25, 90], [0, 35]])
    }

# This will line up the top of the head to the same place on all images
# This will help with hairstyle alignment
# It will also TRY to match the width of all the faces
def get_directional_scale(w, h):
    AVERAGE_H_TO_W = 1.385
    MAX_RATIO = 1.5
    
    ratio = h / w

    # Makes sure the box doesnt get to thin, so does calcuations on a thicker box
    if ratio >= MAX_RATIO:
        w = h / MAX_RATIO
        ratio = MAX_RATIO

    # Gets target size
    target_size = int(SIZE_FACE_MULT * AVERAGE_H_TO_W * w)
    # makes it divisible by 2
    target_size += target_size % 2

    
    average_ratio_h = w * AVERAGE_H_TO_W
    # Face top will be aligned to this position
    average_ratio_y0 = int((target_size - average_ratio_h) / 2)

    # Align face to the right position
    bottom_extend = average_ratio_y0 + math.ceil(h / 2)
    top_extend = target_size - (average_ratio_y0 + math.ceil(h / 2))
    
    # Craete results dictionary
    results = {
        "bottom": bottom_extend,
        "top": top_extend,
        "left": int(target_size / 2),
        "right": int(target_size / 2)
    }

    return results


def crop_image(
    img,
    detect_model :detection_model,
    res_check=True,
    visualize=False,
    bottom_extend=False,
    max_faces=None
    ):
    # Detect faces
    faces = detect_model.inference(img)

    # No face detected
    if len(faces) == 0:
        print("No faces detected... Rejected")
        return None, None
    
    if max_faces is not None and len(faces) > max_faces:
        print("To many faces detected... Rejected")
        return None, None
    
    bounded_images = []
    angle_images = []
    
    for x, y, w, h in faces:  
        
        mid_x, mid_y = int(x + w/2), int(y + h/2)
        max_size = int(max(w,h) / 2)
        
        total_res = int(max_size * SIZE_FACE_MULT * 2)
        
        # Resolution checker
        if res_check and total_res < MINIMUM_QUALITY:
            print(f"{total_res} Resolution wasnt big enough... Rejected")
            continue
        if visualize:
            print("Resolution: ", max_size * SIZE_FACE_MULT * 2)
            print("Face Resolution: ", max_size * 2)
        
        directoinal_scale = get_directional_scale(w, h)
        
        bounds =  (
            mid_x - directoinal_scale["left"], # left
            mid_x + directoinal_scale["right"], # right
            mid_y - directoinal_scale["bottom"], # bottom 
            mid_y + directoinal_scale["top"] # top
        )
        
        """
        print("Mid:", (mid_x, mid_y))
        print("shape:", img.shape)
        print("Size:", max_size * SIZE_FACE_MULT)
        """
        
        l, r, b, t = bounds 
        
        l_bounds, r_bounds, b_bounds, t_bounds = (
            max(0,-1*l), 
            max(0,r - img.shape[1]),
            max(0,-1*b),
            max(0,t - img.shape[0]),
            )
        
        if bottom_extend and t_bounds > 0:
            print("Bottom doesnt reach... Rejected")
            continue
            
        
        bounded_image = img[
            max(0, b): min(img.shape[0], t),
            max(0, l): min(img.shape[1], r)
            ]
        
        bounded_image = cv2.copyMakeBorder(
            bounded_image, 
            b_bounds, # bottom
            t_bounds, #top
            l_bounds,  # left
            r_bounds, # right
            cv2.BORDER_CONSTANT #borderType
            )
        
        
        face_center_x, face_center_y = directoinal_scale["left"], directoinal_scale["bottom"]
        if visualize:
            # Draw bounding boxes
            face_start_x, face_start_y  = (int(face_center_x - w /2), int(face_center_y - h /2))
            cv2.rectangle(bounded_image, (face_start_x, face_start_y), (face_start_x + w, face_start_y + h), (255, 0, 0), 2)
           
            
        
        # Cropped Image for angle detection
        angle_size = int(ANGLE_FACE_MULT * max_size)
        angle_image = bounded_image[
            face_center_y - angle_size : face_center_y + angle_size,
            face_center_x - angle_size : face_center_x + angle_size
            ].copy()
        
        bounded_images.append(bounded_image)
        angle_images.append(angle_image)
    
    if len(bounded_images) == 0:
        bounded_images, angle_images = None, None
    
    return bounded_images, angle_images
        

def clip_cleaner(
    img: np.ndarray, 
    clip_model
    ):
    probs_dic = clip_model.inference(img)
    keep_image = probs_dic["Quality"][0, 0] >= 0.5 and probs_dic["Human"][0, 0] >= 0.5
    if "Advertisement" in probs_dic:
        keep_image = keep_image and probs_dic["Advertisement"][0, 0] > 0.4
    return keep_image, probs_dic

def clean_img(model, path, detect_model: detection_model,
              visualize=False,  # printing and displayingh with matplotlib
              body_parser: body_model=None,  # Body segmentation mode
              clip_model=None,  # Clip model
              res_check=False,  # Only accepts a minimum resolution
              save_path=None,  # Save path
              bottom_extend=False,  # Image must reach the bottom when cropped
              max_faces=None,  # Max faces in an image
              body_width_contain=True,  # Whole body width must be contained inside the image bounds
              reject_greyscale=True,  # Rejects greyscale images
              downscale_big=True,  # Downscales Images that are to big
              get_direction=False
              ):
    
    try:
        raw_img = cv2.imread(path)
    except:
        print("Image errored on Cv2 load... Rejected")
        return None, None 
    
    if raw_img is None:
        print("Image failed to load, image is None... Rejected")
        return None, None 
    
    if reject_greyscale:
        greyscale = np.array_equal(raw_img[:,:, 0], raw_img[:,:, 1]) and np.array_equal(raw_img[:,:, 1], raw_img[:,:, 2])
        
        if greyscale:
            print("Images is greyscale... Rejected")
            return None, None
    
    # makes sure the width of the body is contained in the image
    if body_parser is not None and body_width_contain:
        segmentation, _ = body_parser.inference(raw_img)
        segmentation = segmentation[0]
        
        test_area = np.stack([segmentation[:, :5], segmentation[:, -5:]], axis=1)
        
        invlaid_idxs = np.where(test_area != 0)
        
        if invlaid_idxs[0].shape[0] > 0:
            print("Body width is not contained in the image... Rejected")
            return None, None 
        
        bottom_test = segmentation[:5]
        
        invlaid_idxs = np.where(bottom_test != 0)
        if invlaid_idxs[0].shape[0] > 0:
            print("Body top is not contained in the image... Rejected")
            return None, None

    
    if clip_model is not None:
        keep_image, _ = clip_cleaner(raw_img, clip_model)

        if not keep_image:
            print("Clip rejected image... Rejected")
            return None, None
    
    imgs, angle_images = crop_image(raw_img, detect_model, visualize=visualize, res_check=res_check, bottom_extend=bottom_extend, max_faces=max_faces)
    # If image isn't valid then return
    if imgs is None:
        return None, None
    
    imgs_final = []
    directions = []
    
    for i in range(len(imgs)):
        img = imgs[i]
        angle_image = angle_images[i]

        # Finds valid angle that hte image fits
        valid_direction = None
        if model is not None:
            pitch, yaw, roll = model.predict(angle_image)
            if visualize:
                print(pitch, yaw, roll)
            
            # Checks the validity of the picture face angle based on the given direction
            for direction in ALLOWED_ANGLES:
                if abs(pitch[0]) < ALLOWED_ANGLES[direction][0, 0] or abs(pitch[0]) > ALLOWED_ANGLES[direction][0, 1]:
                    continue
                if abs(yaw[0]) < ALLOWED_ANGLES[direction][1, 0] or abs(yaw[0]) > ALLOWED_ANGLES[direction][1, 1]:
                    continue
                if abs(roll[0]) < ALLOWED_ANGLES[direction][2, 0] or abs(roll[0]) > ALLOWED_ANGLES[direction][2, 1]:
                    continue
                valid_direction = direction
                break
        
        if downscale_big:
            if img.shape[0] > MAX_QUALITY:
                img = cv2.resize(img, (MAX_QUALITY, MAX_QUALITY))
        
        # Visualize
        if visualize:
            model.draw_axis(img, yaw, pitch, roll)
            model.draw_axis(angle_image, yaw, pitch, roll)
            
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
        
        print("Accepted")
        
        if save_path is not None:
            cv2.imwrite(save_path, img)
        
        imgs_final.append(img)
        directions.append(valid_direction)
        
    return imgs_final, directions


def clean_raw_image(raw_img,
                    root_clean_dir,
                    root_raw_dir,
                    mode="hair",
                    delete_raw=True,
                    model:SixDRepNet=None,
                    detect_model:detection_model=None,
                    body_parser: body_model=None,
                    clip_model=None,
                    ):
    
    if detect_model is None:
        return None 
    
    if mode == "hair":
        cleaned_imgs, directions = clean_img(model, raw_img, detect_model=detect_model, res_check=True, get_direction=True)
    elif mode == "body":
        cleaned_imgs, directions = clean_img(
            model, raw_img, 
            detect_model=detect_model, 
            body_parser=body_parser, 
            clip_model=clip_model,
            res_check=True, 
            bottom_extend=True,
            max_faces=1
        )

    # Removes the raw img after it is cleaned
    if delete_raw:
        os.remove(raw_img)
                
    # If image was not accepted continue
    if cleaned_imgs is None or directions is None:
        return []
    
    cleaned_pths = []
    
    for i in range(len(cleaned_imgs)):
        cleaned = cleaned_imgs[i]
        direction = directions[i]
            
        if mode == "hair" and direction is None:
            continue
            
        # Gets the relative path to raw root directory
        pth_list = get_file_path(raw_img, root_raw_dir[2:])
        # Gets the images save path
        if mode == "hair":
            clean_dir = os.path.join(root_clean_dir, pth_list, direction)
        else:
            clean_dir = os.path.join(root_clean_dir, pth_list)
            
        if not os.path.isdir(clean_dir):
            os.makedirs(clean_dir)
            
        # Creates the name
        base_split = os.path.basename(raw_img).split(".")
        new_name = ".".join(base_split[:-1]) + str(i) + "." + base_split[-1]
        final_pth = os.path.join(clean_dir, new_name)
        
        # Save the cleaned image
        cv2.imwrite(final_pth, cleaned)
        
        cleaned_pths.append(final_pth)
    
    return cleaned_pths


def raw_body_image_downloader(clean_queue):
    root_dir = Constants.RAW_BODY_IMAGES_DIR
    completed_download_file = Constants.FINIHSED_BODY_RAW_DOWNLOAD

    rel_base = root_dir.split("/")[-1] + "/"

    folders_to_anaylze = []
    
    # Adds folders on local
    folders_to_anaylze += os.listdir(root_dir)
    
    # Adds folders in AWS S3
    s3_folders = get_download_folders(rel_base)
    for i in range(len(s3_folders)):
        s3_folders[i] = s3_folders[i].split("/")[-1][:-4]
    folders_to_anaylze += s3_folders

    # Removes duplicates
    folders_to_anaylze = set(folders_to_anaylze)

    # Creates file if it doesnt exist
    if not os.path.isfile(completed_download_file):
        file = open(completed_download_file, 'w')
        file.close()

    already_done = []
    # Instanties the finihsed folders from file
    with open(completed_download_file, 'r') as file:
        for x in file.readlines():
            line = x.strip()
            if len(line) == 0:
                continue
            already_done.append(line.replace("\\", "/").split("/")[-1])
    already_done = set(already_done)

    folders_to_anaylze -= set(already_done)

    file_lock = Lock()
    
    for folder in folders_to_anaylze:
        while len(find_images(Constants.RAW_BODY_IMAGES_DIR)) > 5000:
            time.sleep(60)

        folder_path = os.path.join(Constants.RAW_BODY_IMAGES_DIR, folder)
        key = rel_base + folder + ".zip"
        print("Downloading", key)
        try:
            zip_process_thread = download_aws_folder(folder_path, key, finished_download_file=completed_download_file, file_lock=file_lock)
            zip_process_thread.join()
    
            for img_name in os.listdir(folder_path):
                img_pth = os.path.join(folder_path, img_name)
                clean_queue.put(img_pth)
        except Exception as e:
            print("Download Failed:", e)


# Preprocessor
def Preprocess(clean_queue, accept_queue,
               root_clean_dir, root_raw_dir,
               mode="hair", delete_raw=True):

    torch.cuda.empty_cache()

    # Create model
    # Weights are automatically downloaded
    model = SixDRepNet() if mode == "hair" else None
    
    detect_model = detection_model()
    
    if mode == "body":
        body_parser = body_model()
        clip_model = Clip()
    else:
        body_parser = None
        clip_model = HairClip()
    
    total = 0
    accepted = 0
    rejected = 0
    
    # Loop until errors out from timeout (nothing more to clean)
    while True:
        # End early if wants to exit
        if getStop():
            return
        
        # Gets image and cleans it
        try:
            raw_img = clean_queue.get(timeout=120)
        except Exception as e:
            return
            
        if not os.path.isfile(raw_img):
            continue
        
        final_pths = clean_raw_image(
            raw_img, root_clean_dir, root_raw_dir, mode=mode,
            model=model,
            detect_model=detect_model,
            body_parser=body_parser,
            clip_model=clip_model
            )
        
        total += 1
        
        if len(final_pths) == 0:
            rejected += 1
        else:
            accepted += 1
        
        if total % 50 == 0:
            accepted_perc = accepted / total
            print(f"\n\nACCETPED {accepted}/{total} = {accepted_perc}\n\n")
            
        # Adds to the queue for acceptance
        if accept_queue is not None:
            for final_pth in final_pths:
                accept_queue.put(final_pth)


if __name__ == "__main__":
    # Create model
    # Weights are automatically downloaded
    model = SixDRepNet()
    
    body_parser = body_model()
    
    detect_model = detection_model()

    clip_model = Clip()

    test_dir = "data/clean_images/dreads portrait"
    names = os.listdir(test_dir)
    names.sort()

    amount = 0
    
    for img_name in names:
        img_pth = os.path.join(test_dir, img_name)
        if not os.path.isfile(img_pth):
            continue

        img = cv2.imread(img_pth)

        x0, y0, x1, y1 = 0, 0, img.shape[1], img.shape[0]

        x_line = np.zeros((img.shape[0], 3))
        y_line = np.zeros((img.shape[1], 3))
        
        for y in range(0, img.shape[0]):
            if np.array_equal(img[y, :], y_line):
                y0 = y
            else:
                break

        for y in range(img.shape[0] - 1, y0, -1):
            if np.array_equal(img[y, :], y_line):
                y1 = y
            else:
                break
        for x in range(0, img.shape[1]):
            if np.array_equal(img[:, x], x_line):
                x0 = x
            else:
                break

        for x in range(img.shape[1] - 1, x0, -1):
            if np.array_equal(img[:, x], x_line):
                x1 = x
            else:
                break

        keep_image, probs_dic = clip_cleaner(img[y0:y1, x0:x1], clip_model)

        print(img_pth)
        print("Quality:", probs_dic["Quality"], "\nHuman:", probs_dic["Human"], "\n")
        amount += 1
            
        if amount > 20:
            break
    
    exit()
    
    """
    test_dir = "data/raw_images/street fashion portrait"
    for img_name in os.listdir(test_dir):
        img_pth = os.path.join(test_dir, img_name)
        clean_img(model, img_pth, detect_model=detect_model, visualize=True, res_check=True, bottom_extend=True)
    """
    
    
    
