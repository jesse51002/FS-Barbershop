import os
import sys

import PIL.Image
sys.path.insert(0, './')
from PIL import Image
from pathlib import Path
import torchvision
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy
import cv2
import math
import time

from args_maker import create_parser
from models.ModelBase import Model

WEIGHT_FILE_NAME = "shape_predictor_68_face_landmarks.dat"
BATCH_SIZE = 1

SIZE_FACE_MULT = 3.5
ANGLE_FACE_MULT = 2

RESOLUTION = 1024


class Preprocessor(nn.Module):
    def __init__(self, opts: dict, background_remover: Model, detection_model: Model):
        """
        Initializes a new instance of the Preprocessor class.
        Parameters:
            opts (object): The options object containing configuration settings.
            background_remover (object): The background remover object
            keypoint_model (object): The keypoint model object
        """
        
        super(Preprocessor, self).__init__()
        self.opts = opts
        self.background_remover = background_remover
        self.detection_model = detection_model
    
    def set_opts(self, opts: dict) -> None:
        self.opts = opts

    def preprocess_imgs(self, imgs: list[str], target_color: int = 255) -> None:
        """
        Preprocesses a list of images.
        
        Args:
            imgs (List[str]): A list of paths to the input images.
            target_color (int, optional): The target color for the background removal. Defaults to 155.
        
        Returns:
            None
        
        Raises:
            AssertionError: If the file type of an image is not "jpeg", "jpg", or "png".
        
        Description:        
            * Preprocesses a list of images
            * For each image, it removes the background using the background remover and saves the result in the unprocessed directory
            * For each image, it aligns the face using the keypoints and saves the result in the input directory
            * Saves the images in the input directory
        """
        
        torch.cuda.set_device(self.opts.device[0])
        
        if not os.path.isdir(self.opts.unprocessed):
            os.makedirs(self.opts.unprocessed)

        if not os.path.isdir(self.opts.input_dir):
            os.makedirs(self.opts.input_dir)
        
        imgs = imgs.copy()
        
        i = 0
        while i < len(imgs):
            im = imgs[i]
            base_name_split = os.path.basename(im).split(".")
            
            im_type = base_name_split[1]
            assert im_type in ["jpeg", "jpg", "png"], f"Invlaid file type: {im}"

            im_name = base_name_split[0]
            output_img_pth = os.path.join(self.opts.input_dir, im_name + ".png")
            
            if os.path.isfile(output_img_pth):
                imgs.pop(i)
                i -= 1
                if not self.opts.disable_progress_bar:
                    print("Alignment already done, skipping", im)
            i += 1
        
        if len(imgs) == 0:
            return

        print("Preprocessing:", imgs)
        batched_input = torch.zeros((len(imgs), 3, RESOLUTION, RESOLUTION))
        
        for i, img_path in enumerate(imgs):
            image = cv2.imread(img_path)
            
            np_image = self.crop_image(image)
            
            np_image = cv2.resize(
                    np_image,
                    (RESOLUTION, RESOLUTION),
                    interpolation=cv2.INTER_CUBIC if np_image.shape[0] < RESOLUTION else cv2.INTER_AREA
                )
            
            torch_img = torch.tensor(np_image[:, :, ::-1].copy()).permute(2, 0, 1)
            batched_input[i] = torch_img
        
        for i in range(0, len(imgs), BATCH_SIZE):
            cur_batched_input = batched_input[i: min(i+BATCH_SIZE, len(imgs))]
            cur_imgs = imgs[i: min(i+BATCH_SIZE, len(imgs))]
            
            # Gets the human matting segmentaion mask
            cur_output_imgs, _ = self.background_remover.inference(cur_batched_input, target_color=target_color)
            cur_output_imgs = cur_output_imgs[:, ::-1].transpose((0, 2, 3, 1))

            # Aligns face based of model outputs
            for img_i in range(len(cur_imgs)):
                im = cur_imgs[img_i]
                im_stem = os.path.basename(im).split(".")[0]
                output_pth = os.path.join(self.opts.input_dir, im_stem + ".png")
                cv2.imwrite(output_pth, cur_output_imgs[img_i])
                    
    def get_directional_scale(self, w, h):
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

    def crop_image(self, img, bottom_extend=False):
        model_input = torch.tensor(img[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float().to(self.opts.device[0]) 
        
        # Detect faces
        faces = self.detection_model.inference(model_input)
                
        x, y, x1, y1 = faces['rects'][0]
        w = x1 - x
        h = y1 - y
            
        mid_x, mid_y = int(x + w/2), int(y + h/2)
            
        directoinal_scale = self.get_directional_scale(w, h)
            
        bounds = (
            mid_x - directoinal_scale["left"], # left
            mid_x + directoinal_scale["right"], # right
            mid_y - directoinal_scale["bottom"], # bottom 
            mid_y + directoinal_scale["top"] # top
        )
            
        l, r, b, t = bounds
            
        l_bounds, r_bounds, b_bounds, t_bounds = (
            max(0,-1*l), 
            max(0,r - img.shape[1]),
            max(0,-1*b),
            max(0,t - img.shape[0]),
            )
            
        if bottom_extend and t_bounds > 0:
            raise Exception("Bottom doesnt reach")
                
            
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
            cv2.BORDER_CONSTANT, #borderType
            value=255, # Color
        )
                                
        return bounded_image
        

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args([])

    from models.FacerParsing import FacerDetection
    from models.p3m_matting.inference import human_matt_model
    
    face_detector = FacerDetection()
    background_remover = human_matt_model(device=args.device[0])
    
    preprocessor = Preprocessor(args, detection_model=face_detector, background_remover=background_remover)
    
    unprocessed_dir = "input/unprocessed"

    img_pths = [os.path.join(unprocessed_dir, x) for x in os.listdir(unprocessed_dir)]
    ipynb_bitch = os.path.join(unprocessed_dir, ".ipynb_checkpoints")
    if ipynb_bitch in img_pths:
        img_pths.remove(ipynb_bitch)

    # img_pths = ["input/unprocessed/pink-blonde.jpg"]

    start = time.time()
    preprocessor.preprocess_imgs(img_pths)
    print("Took ", time.time() - start)