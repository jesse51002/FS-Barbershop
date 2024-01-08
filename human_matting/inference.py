import sys
sys.path.insert(0,'./')
sys.path.insert(0,'./human_matting')

import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from mat_model.model import HumanSegment, HumanMatting
import matting_utils
from  utils.seg_utils import save_original_mask

CHECKPOINT = "human_matting/SGHM-ResNet50.pth"

model = HumanMatting(backbone='resnet50')
model = nn.DataParallel(model).cuda().eval()
model.load_state_dict(torch.load(CHECKPOINT))

pil_to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

infer_size = 1280

def remove_background(img, target_color=155):
    h,w = None, None
    if isinstance(img, str):
        img = cv2.imread(img)[:,:, ::-1].copy()
        h, w = img.shape[:2]
        img = pil_to_tensor(img).permute((2,0,1)).unsqueeze(0).float().cuda()
    elif isinstance(img, np.ndarray):
        print(img.shape)
        h, w = img.shape[:2]
        img = pil_to_tensor(img[:,:, ::-1].copy()).unsqueeze(0).float().cuda()
    elif isinstance(img, Image.Image):
        h = img.height
        w = img.width
        img = pil_to_tensor(np.array(img)).unsqueeze(0).float().cuda()
    elif isinstance(img, torch.Tensor):
        h, w = img.shape[-2:]
        img = img.flip(dims=[-3]).cuda()
    else:
        raise TypeError
    
    rh, rw = None, None
    if w >= h:
        rh = infer_size
        rw = int(w / h * infer_size)
    else:
        rw = infer_size
        rh = int(h / w * infer_size)
    rh = rh - rh % 64
    rw = rw - rw % 64    

    
    # print(img.shape, (h,w), (rh,rw))

    input_tensor = F.interpolate(img, size=(rh, rw), mode='bilinear')
    with torch.no_grad():
        pred = model(input_tensor)
    
    # progressive refine alpha
    alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
    pred_alpha = alpha_pred_os8.clone().detach()
    weight_os4 = matting_utils.get_unknown_tensor_from_pred(pred_alpha, rand_width=30, train_mode=False)
    pred_alpha[weight_os4>0] = alpha_pred_os4[weight_os4>0]
    weight_os1 = matting_utils.get_unknown_tensor_from_pred(pred_alpha, rand_width=15, train_mode=False)
    pred_alpha[weight_os1>0] = alpha_pred_os1[weight_os1>0]

    pred_alpha = pred_alpha.repeat(1, 3, 1, 1)
    pred_alpha = F.interpolate(pred_alpha, size=(h, w), mode='bilinear')
    alpha_np = pred_alpha.data

    # output segment
    pred_segment = pred['segment']
    pred_segment = F.interpolate(pred_segment, size=(h, w), mode='bilinear')
    segment_np = pred_segment.data

    backremoved = torch.multiply(img.flip(dims=[-3]), alpha_np).flip(dims=[-3]) * 255  
    base = torch.ones_like(img) * target_color / 255
    distance = img - base
    new_img = (base + distance * alpha_np) * 255
    
    return new_img


if __name__ == "__main__":
    face_dir = "./unprocessed"
    accepted_imgs = ["png", "jpg", "jpeg"]
    
    torch.no_grad()
    for img_name in os.listdir(face_dir):
        if img_name.split(".")[-1] not in accepted_imgs:
            continue

        print(img_name)
        img_pth = os.path.join(face_dir, img_name)
        unscaled_img = torch.tensor(cv2.imread(img_pth)[:,:,::-1].copy()).permute((2,0,1))
        unscaled_img = unscaled_img.unsqueeze(0).float().to("cuda:0")
        img = unscaled_img / 255
        print("Img shape:",img.shape)
        output_img = remove_background(img).detach().cpu().numpy()[0]
        print("output:", output_img.shape)
        cv2.imwrite(f"./output/background_removed/{img_name}", output_img.transpose(1,2,0))

