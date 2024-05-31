import os
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from models.FacerParsing import facer_to_bisnet
from utils.seg_utils import expand_face_mask, vis_seg

BATCH_SIZE = 4


class SegMaker(nn.Module):
    def __init__(self, opts, facer=None, background_remover=None, keypoint_model=None):
        super(SegMaker, self).__init__()
        self.opts = opts
        self.facer = facer
        self.background_remover = background_remover
        self.keypoint_model = keypoint_model

    def set_opts(self, opts):
        self.opts = opts

    def create_segmentations(self, imgs):
        imgs = imgs.copy()
        
        i = 0
        while i < len(imgs):
            img_path = imgs[i]
            name = os.path.basename(img_path).split(".")[0]
            save_dir = os.path.join(self.opts.output_dir, 'masks')
            save_pth = os.path.join(save_dir, name) + "_mask.npz"
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

        # This is the base for concactinating the sub batches
        human_segs = torch.zeros((0, 512, 512)).to(self.background_remover.device)
        keypoints = torch.zeros((0, 68, 2)).to(self.keypoint_model.device)
        seg_targets = torch.zeros((0, 512, 512)).to(self.facer.device)
    
        # Inferences sub batches
        for i in range(0, len(imgs), BATCH_SIZE):
            cur_batched_input = batched_input[i: min(i+BATCH_SIZE, len(imgs))]

            # Gets the human matting segmentaion mask
            _, cur_human_seg = self.background_remover.inference(cur_batched_input)

            # Gets the keypoints results
            cur_keypoints = self.keypoint_model.inference(cur_batched_input)['alignment']

            # Gets face parsing results
            cur_seg_targets = facer_to_bisnet(self.facer.inference(cur_batched_input)[0])
            
            human_segs = torch.cat([human_segs, cur_human_seg], dim=0)
            keypoints = torch.cat([keypoints, cur_keypoints], dim=0)
            seg_targets = torch.cat([seg_targets, cur_seg_targets], dim=0)

        seg_targets = seg_targets.float().data.cpu().numpy()
        human_segs = human_segs.data.cpu().numpy()
        keypoints = keypoints
        
        equation_results = self.keypoint_model.poly_interpolate(keypoints)

        for i in range(len(imgs)):
            img_path = imgs[i]
            face_seg = expand_face_mask(seg_targets[i], human_segs[i])

            # ---- Saves Images
            save_dir = os.path.join(self.opts.output_dir, 'masks')
            os.makedirs(save_dir, exist_ok=True)
            
            name = os.path.basename(img_path).split(".")[0]

            np.savez(
                os.path.join(save_dir, name) + "_mask.npz",
                mask=face_seg, 
                left_points=equation_results[i]["left"],
                right_points=equation_results[i]["right"],
                brows_points=equation_results[i]["brows"]
            )
            
            rgb_img = vis_seg(face_seg)
            vis_img = Image.fromarray(rgb_img)
            vis_img.save(os.path.join(save_dir, name) + ".png")
        
