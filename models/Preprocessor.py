import os
import sys
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

WEIGHT_FILE_NAME = "shape_predictor_68_face_landmarks.dat"
BATCH_SIZE = 3


class Preprocessor(nn.Module):
    def __init__(self, opts, background_remover=None, keypoint_model=None):
        super(Preprocessor, self).__init__()
        self.opts = opts
        self.background_remover = background_remover
        self.keypoint_model = keypoint_model
    
    def set_opts(self, opts):
        self.opts = opts

    def preprocess_imgs(self, imgs, target_color=155):
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
        target_size = 2048
        batched_input = torch.zeros((len(imgs), 3, target_size, target_size))
        
        for i, img_path in enumerate(imgs):
            image = Image.open(img_path)

            large_size = max(image.size)

            # Scales down if needed
            if large_size > target_size:
                scale_amount = target_size / large_size
                image = image.resize((int(scale_amount * image.size[0]), int(scale_amount * image.size[1])), PIL.Image.LANCZOS)

            np_image = np.array(image.convert('RGB'))

            h_diff = max(0, target_size - np_image.shape[0])
            w_diff = max(0, target_size - np_image.shape[1])
            
            np_image = cv2.copyMakeBorder(
                np_image,
                math.floor(h_diff/2),
                math.ceil(h_diff/2),
                math.floor(w_diff/2),
                math.ceil(w_diff/2),
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0])
            
            if sum(np_image.shape[:2]) != target_size * 2:
                np_image = cv2.resize(
                    np_image,
                    (target_size, target_size),
                    interpolation=cv2.INTER_LINEAR
                )
            
            torch_img = torch.tensor(np_image).permute(2, 0, 1)
            batched_input[i] = torch_img

        output_imgs = np.zeros((0, 3, target_size, target_size))
        keypoints = torch.zeros((0, 68, 2)).to(self.keypoint_model.device)
        
        for i in range(0, len(imgs), BATCH_SIZE):
            cur_batched_input = batched_input[i: min(i+BATCH_SIZE, len(imgs))]
            # Gets the human matting segmentaion mask
            cur_output_imgs, _ = self.background_remover.inference(cur_batched_input, target_color=target_color)
            # Gets the keypoints results
            cur_keypoints = self.keypoint_model.inference(cur_batched_input)['alignment']

            output_imgs = np.concatenate([output_imgs, cur_output_imgs], axis=0)
            keypoints = torch.cat([keypoints, cur_keypoints], dim=0)

        # Aligns face based of model outputs
        for img_i in range(len(imgs)):
            faces = self.align_face(output_imgs[img_i], keypoints[img_i].data.cpu().numpy())
    
            for i, face in enumerate([faces[0]]):
                factor = 1024//self.opts.size
                assert self.opts.size*factor == self.opts.size
                face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
                face_tensor_lr = face_tensor[0].cpu().detach().clamp(0, 1)
                face = torchvision.transforms.ToPILImage()(face_tensor_lr)
                if factor != 1:
                    face = face.resize((self.opts.size, self.opts.size), PIL.Image.LANCZOS)

                im = imgs[img_i]
                im_stem = os.path.basename(im).split(".")[0]
                
                face.save(Path(self.opts.input_dir) / (im_stem + ".png"))
    
    def align_face(self, img, keypoints):
        """
        :param filepath: str
        :return: list of PIL Images
        """
        
        lm = keypoints
        
        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]   # np.expand_dims(lm[36], 0)  #  # left-clockwise
        lm_eye_right = lm[42: 48]  # np.expand_dims(lm[45], 0)  #   # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise
    
        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg
        
        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        img = PIL.Image.fromarray(img.transpose((1, 2, 0)).astype(np.uint8))
            
        output_size = 1024
        # output_size = 256
        transform_size = 4096
        enable_padding = True
    
        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink
    
        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]
    
        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]
    
        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.LANCZOS)
   
        return [img]
        

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args([])

    from models.FacerParsing import FacerKeypoints, FacerDetection
    from models.p3m_matting.inference import human_matt_model
    
    face_detector = FacerDetection()
    keypoint_model = FacerKeypoints(face_detector=face_detector, device=args.device[0])
    background_remover = human_matt_model(device=args.device[0])
    
    preprocessor = Preprocessor(args, keypoint_model=keypoint_model, background_remover=background_remover)
    
    unprocessed_dir = "input/unprocessed"

    img_pths = [os.path.join(unprocessed_dir, x) for x in os.listdir(unprocessed_dir)]
    ipynb_bitch = os.path.join(unprocessed_dir, ".ipynb_checkpoints")
    if ipynb_bitch in img_pths:
        img_pths.remove(ipynb_bitch)

    # img_pths = ["input/unprocessed/pink-blonde.jpg"]

    start = time.time()
    preprocessor.preprocess_imgs(img_pths)
    print("Took ", time.time() - start)