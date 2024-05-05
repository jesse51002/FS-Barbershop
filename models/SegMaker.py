import os
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from models.FacerParsing import facer_to_bisnet
from utils.seg_utils import expand_face_mask, vis_seg


class SegMaker(nn.Module):
    def __init__(self, opts, facer=None, background_remover=None, keypoint_model=None):
        super(SegMaker, self).__init__()
        self.opts = opts
        self.facer = facer
        self.background_remover = background_remover
        self.keypoint_model = keypoint_model

    def create_segmentations(self, img_path1, img_path2, img_path3):
        imgs = list(set([img_path1, img_path2, img_path3]))

        i = 0
        while i < len(imgs):
            img_path = imgs[i]
            name = os.path.basename(img_path).split(".")[0]
            save_dir = os.path.join(self.opts.output_dir, 'masks')
            save_pth = os.path.join(save_dir, name) + ".npy"
            if os.path.isfile(save_pth):
                imgs.pop(i)
                i -= 1
            i += 1
        
        if len(imgs) == 0:
            return

        print("Segmenting:", imgs)
        
        batched_input = torch.zeros((len(imgs), 3, 1024, 1024))
        
        for i, img_path in enumerate(imgs):
            image = Image.open(img_path)
            np_image = np.array(image.convert('RGB'))
            torch_img = torch.tensor(np_image).permute(2, 0, 1)
            batched_input[i] = torch_img

        batched_input = F.interpolate(batched_input, size=(512, 512), mode='nearest')

        # print(batched_input.shape)
        # uses high quality segmentation for segmenting
        seg_targets = facer_to_bisnet(self.facer.inference(batched_input)[0]).float().detach().cpu().numpy()

        # Gets the human matting segmentaion mask
        human_segs = self.background_remover.inference(batched_input)[1].detach().cpu().numpy()

        # Gets the keypoints results
        keypoints = self.keypoint_model.inference(batched_input)
        right_coef, left_coef, y_range = self.keypoint_model.create_poly_equation(keypoints)
        equation_results = self.keypoint_model.inference_poly_eq(right_coef, left_coef, y_range)

        for i in range(len(imgs)):
            img_path = imgs[i]
            face_seg = expand_face_mask(seg_targets[i], human_segs[i])

            # ---- Saves Images
            save_dir = os.path.join(self.opts.output_dir, 'masks')
            os.makedirs(save_dir, exist_ok=True)
            
            name = os.path.basename(img_path).split(".")[0]
            # Saves face parsing
            np.save(os.path.join(save_dir, name) + ".npy", face_seg)
            # Saves face keypoints
            np.save(os.path.join(save_dir, name) + "_left_points.npy", equation_results[i]["left"])
            np.save(os.path.join(save_dir, name) + "_right_points.npy", equation_results[i]["right"])
            
            rgb_img = vis_seg(face_seg)
            vis_img = Image.fromarray(rgb_img)
            vis_img.save(os.path.join(save_dir, name) + ".png")
        
