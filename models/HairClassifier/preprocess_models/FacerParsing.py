import os

import time
import torch
from PIL import Image
import numpy as np

import facer

FACER_CLASSES = {
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


class FacerDetection:
    def __init__(self, device="cuda"):
        self.name = "FacerDetection"
        self.device = device
        self.face_detector_model = facer.face_detector('retinaface/mobilenet', device=self.device)
    
    def inference(self, imgs: torch.Tensor, keep_one_img: bool = True) -> dict:
        """
        Input:
        imgs (torch tensor) (RGB): b x c x h x w
        keep_one_img (bool): If True, keeps only the highest confidence face. If False, keeps all faces.

        Output:
        Tuple (
            seg_results (torch.Tensor): nfaces x h x w
            seg_logits (torch.Tensor): nfaces x nclasses x h x w,
        )
        """

        imgs = imgs.to(self.device)

        with torch.inference_mode():
            faces = self.face_detector_model(imgs)

        if keep_one_img:
            predicted_count = faces["image_ids"].shape[0]
            
            keep_idxs = [-1] * imgs.shape[0]
            keep_top_perc = [-1] * imgs.shape[0]
            
            for i in range(predicted_count):
                img_idx = faces["image_ids"][i]
                
                score = faces["scores"][i]
    
                if score > keep_top_perc[img_idx]:
                    keep_idxs[img_idx] = i
                    keep_top_perc[img_idx] = score
    
            bool_selector = torch.zeros((predicted_count)).bool()
            for i in keep_idxs:
                assert i >= 0, f"No faces were found in image {i} face detection"
                bool_selector[i] = True
    
            for key in faces:
                faces[key] = faces[key][bool_selector]
        
        return faces


class FacerModel:
    
    def __init__(self, face_detector: FacerDetection, device="cuda"):
        self.name = "FacerFaceParsing"
        self.device = device
        self.segmentation_model = facer.face_parser('farl/lapa/448', device=self.device)
        self.face_detector = face_detector
    
    def inference(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


"""
if __name__ == "__main__":
    face_detector = FacerDetection()
    
    model = FacerModel(face_detector=face_detector)

    results, __ = model.inference(torch_img)
    results = results.detach().cpu()[0]

    converted_results = facer_to_bisnet(results)
"""


