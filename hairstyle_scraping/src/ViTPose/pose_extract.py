import sys
sys.path.insert(0,'./src')
sys.path.insert(0,'./src/ViTPose')

# This error  subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1.
# This has the fix https://github.com/HRNet/HRNet-Semantic-Segmentation/issues/39
# Then restart your vscode

import os
from Model import Model

import cv2
import time
import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download

# from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)


class pose_model(Model):
    MODEL_DICT = {
        'ViTPose-B (single-task train)': {
            'config':
            'src/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
            'model': 'models/vitpose-b.pth',
        },
        'ViTPose-L (single-task train)': {
            'config':
            'src/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
            'model': 'models/vitpose-l.pth',
        },
        'ViTPose-B (multi-task train, COCO)': {
            'config':
            'src/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
            'model': 'models/vitpose-b-multi-coco.pth',
        },
        'ViTPose-L (multi-task train, COCO)': {
            'config':
            'src/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
            'model': 'models/vitpose-l-multi-coco.pth',
        },
    }

    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_name = 'ViTPose-B (multi-task train, COCO)'
        self.model = self._load_model(self.model_name)
        
        # self.det_model = person_det_model()

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)
            
    def _load_model(self, name: str) -> nn.Module:
        d = self.MODEL_DICT[name]
        ckpt_path = hf_hub_download('public-data/ViTPose',
                                                    d['model'])
        model = init_pose_model(d['config'], ckpt_path, device=self.device)
        return model

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        out = self.inference(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold,
                                          vis_dot_radius, vis_line_thickness)
        return out, vis

    def inference(
            self,
            image: np.ndarray,
            # det_results: list[np.ndarray],
            box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
        # image = image[:, :, ::-1]  # RGB -> BGR
        
        det_results = np.array([[[0,0, 1024, 1024, 1.0]]]) # self.person_det_model.inference(image)
        
        person_results = process_mmdet_results(det_results, 1)
        
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               person_results=person_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        
        # out[0]["keypoints"][:, :2] = out[0]["keypoints"][:, :2].astype(int)
        return out
    
    def index_to_dic(self, pose2d):
        
        return {
            "nose": pose2d[0],
            "left_eye": pose2d[1],
            "right_eye": pose2d[2],
            "left_ear": pose2d[3],
            "right_ear": pose2d[4],
            "left_shoulder": pose2d[5],
            "right_shoulder": pose2d[6],
            "left_elbow": pose2d[7],
            "right_elbow": pose2d[8],
            "left_hand": pose2d[9],
            "right_hand": pose2d[10],
            "left_waist": pose2d[11],
            "right_waist": pose2d[12],
            "left_knee": pose2d[13],
            "right_knee": pose2d[14],
            "left_foot": pose2d[15],
            "right_foot": pose2d[16],
        }
        
    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: list[np.ndarray],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        #image = image[:, :, ::-1]  # RGB -> BGR
        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness) 
        return vis #[:, :, ::-1]  # BGR -> RGB
         
    
if __name__ == "__main__": 
    model = pose_model()

    img_pth = "./data/clean_images/fitness portrait curly/46684915021_e79b9ea865_o0.jpg"
    img = cv2.imread(img_pth)
    print("Img shape:", img.shape)

    start = time.time()
    raw_output = model.inference(img)

    print()
    positions = raw_output[0]["keypoints"]
    # output_img = output_img.detach().cpu().numpy()[0]
    print("Took:", time.time() - start)
    print("output:", positions)
        
    vis = model.visualize_pose_results(img, raw_output)
    cv2.imwrite("./src/ViTPose/pose_example.jpg", vis)
        