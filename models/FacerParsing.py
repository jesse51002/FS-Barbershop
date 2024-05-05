import os
import sys
sys.path.insert(0, './')

import time
import torch
from PIL import Image
import numpy as np


import facer
from models.ModelBase import Model
from models.face_parsing.classes import CLASSES as bisnet_classes

CLASSES = {
        'background': 0,
        'face': 1,
        'rbrow': 2,
        'lbrow': 3,
        'reye': 4,
        'leye': 5,
        'nose': 6,
        'ulip': 7,
        'mouth': 8,
        'llip': 9,
        'hair': 10,
    }


FACER_TO_BISNET = {
    'background': "background",
    'face': "face",
    'rbrow': "brows",
    'lbrow': "brows",
    'reye': "eyes",
    'leye': "eyes",
    'nose': "nose",
    'ulip': "ulip",
    'mouth': "mouth",
    'llip': "llip",
    'hair': "hair",
}


class FacerDetection(Model):
    def __init__(self, device="cuda"):
        self.device = device
        self.face_detector_model = facer.face_detector('retinaface/mobilenet', device=self.device)
    
    def inference(self, imgs):
        """
        Input:
        imgs (torch tensor) (RGB): b x c x h x w

        Output:
        Tuple (
            seg_results (torch.Tensor): nfaces x h x w
            seg_logits (torch.Tensor): nfaces x nclasses x h x w,
        )
        """

        imgs = imgs.to(self.device)

        with torch.inference_mode():
            faces = self.face_detector_model(imgs)
            
        return faces


class FacerModel(Model):
    def __init__(self, face_detector: FacerDetection, device="cuda"):
        self.device = device
        self.segmentation_model = facer.face_parser('farl/lapa/448', device=self.device)
        self.face_detector = face_detector
    
    def inference(self, imgs):
        """
        Input:
        imgs (torch tensor) (RGB): b x c x h x w

        Output:
        Tuple (
            seg_results (torch.Tensor): nfaces x h x w
            seg_logits (torch.Tensor): nfaces x nclasses x h x w,
        )
        """

        imgs = imgs.to(self.device)

        faces = self.face_detector.inference(imgs)
        
        with torch.inference_mode():
            faces = self.segmentation_model(imgs, faces)

        labels = faces['seg']['label_names']
        labels_dict = {}

        for i, l in enumerate(labels):
            labels_dict[i] = l
        
        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
        seg_results = seg_probs.argmax(dim=1)

        return seg_results, seg_logits


class FacerKeypoints(Model):
    def __init__(self, face_detector: FacerDetection, device="cuda"):
        self.device = device
        self.keypoint_model = facer.face_aligner('farl/ibug300w/448', device=device)
        self.face_detector = face_detector
    
    def inference(self, imgs):
        """
        Input:
        imgs (torch tensor) (RGB): b x c x h x w

        Output:
        Tuple (
            seg_results (torch.Tensor): nfaces x h x w
            seg_logits (torch.Tensor): nfaces x nclasses x h x w,
        )
        """

        imgs = imgs.to(self.device)

        faces = self.face_detector.inference(imgs)
        
        with torch.inference_mode():
            keypoints = self.keypoint_model(imgs, faces)

        return keypoints

    def create_poly_equation(self, keypoints, degree=4):
        left_start = 0
        middle = 8
        right_end = 16

        np_points = keypoints['alignment'].data.cpu().numpy()

        right = []
        left = []
        y_range = np.zeros((np_points.shape[0], 2))

        for i in range(np_points.shape[0]):
            y_range[i, 0] = np_points[i, left_start, 1]
            y_range[i, 1] = np_points[i, middle, 1]
            
            right_points = np_points[i, middle: right_end + 1]
            right_coefs = np.polyfit(right_points[:, 1], right_points[:, 0], degree)
    
            left_points = np_points[i, left_start: middle + 1]
            left_coefs = np.polyfit(left_points[:, 1], left_points[:, 0], degree)

            right.append(right_coefs)
            left.append(left_coefs)
        
        right = np.stack(right, axis=0)
        left = np.stack(left, axis=0)
        
        return right, left, y_range

    def inference_poly_eq(self, r_coef, l_coef, y_range):
        outputs = []

        def InferPoly(y_list, coef):
            p = np.poly1d(coef)

            return p(y_list)
            # print(coef)
            #return sum([coef[p]*y_list**(coef.shape[0] - p - 1) for p in range(coef.shape[0])])
        
        for i in range(y_range.shape[0]):
            
            y_list = np.arange(y_range[i, 0], y_range[i, 1], 1).astype(np.int32)

            r_answers = np.stack([InferPoly(y_list, r_coef[i]), y_list], axis=1).astype(np.int32)
            l_answers = np.stack([InferPoly(y_list, l_coef[i]), y_list], axis=1).astype(np.int32)

            outputs.append({"right": r_answers, "left": l_answers})
        
        return outputs
        

def facer_to_bisnet(facer_seg):
    """
    Input (Facer segmentation):
    seg_results (torch.Tensor): nfaces x h x w

    Output (Unet segmentation):
    seg_output (torch.Tensor): nfaces x h x w
    """

    output_seg = torch.zeros_like(facer_seg)
    
    for key in CLASSES:
        # index facer
        key_idx = CLASSES[key]

        # Goal idx in unet
        new_idx = bisnet_classes[FACER_TO_BISNET[key]]

        # Converts each class
        output_seg = torch.where(facer_seg == key_idx, new_idx, output_seg)

    output_seg = output_seg.to(facer_seg.device)
    
    return output_seg


if __name__ == "__main__":
    # path = "input/face/braids2.png"
    for img_name in ["short_hair.png", "werewolf.png"]:  # os.listdir("input/face/")[:5]:
        path = os.path.join("input/face/", img_name)
        
        if not os.path.isfile(path):
            continue
        
        image = Image.open(path)
        np_image = np.array(image.convert('RGB'))
        torch_img = torch.tensor(np_image).permute(2, 0, 1).unsqueeze(0)
    
        face_detector = FacerDetection()
        keypoint_model = FacerKeypoints(face_detector=face_detector)

        start = time.time()
        keypoints = keypoint_model.inference(torch_img)

        right_coef, left_coef, y_range = keypoint_model.create_poly_equation(keypoints)
        equation_results = keypoint_model.inference_poly_eq(right_coef, left_coef, y_range)

        for i in range(len(equation_results)):
            current_results = equation_results[i]
            for j in range(current_results["left"].shape[0]):
                white = np.array([255, 255, 255])
                np_image[current_results["left"][j, 1]: current_results["left"][j, 1] + 10, current_results["left"][j, 0]: current_results["left"][j, 0] + 10] = white
                np_image[current_results["right"][j, 1]: current_results["right"][j, 1] + 10, current_results["right"][j, 0]: current_results["right"][j, 0] + 10] = white
            
        print("Finsihed in:", time.time() - start)

        for pts in keypoints['alignment']:
            np_image = facer.draw_landmarks(np_image, None, pts.cpu().numpy())

        output_img = Image.fromarray(np_image)
        output_img.save(os.path.join("test_output", "points_" + os.path.basename(path)))
    
    """
    model = FacerModel(face_detector=face_detector)

    results, __ = model.inference(torch_img)
    results = results.detach().cpu()

    converted_results = facer_to_bisnet(results)
    
    from utils.seg_utils import vis_seg
    vis_img = Image.fromarray(vis_seg(results[0]))
    vis_img.save(os.path.join("test_output", os.path.basename(path)))

    vis_img_converted = Image.fromarray(vis_seg(converted_results[0]))
    vis_img_converted .save(os.path.join("test_output", "converted_" + os.path.basename(path)))
    """
    

