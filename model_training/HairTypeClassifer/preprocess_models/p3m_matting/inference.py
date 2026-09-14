import sys
sys.path.insert(0,'./models/p3m_matting/core')

import os
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import pandas as pd
from p3m_util import gen_trimap_from_segmap_e2e
from network import build_model

CHECKPOINT = "models/p3m_matting/models/pretrained/P3M-Net_ViTAE-S_trained_on_P3M-10k.pth"
ARCH = 'vitae'

# build model
model = build_model(ARCH, pretrained=False)

# load ckpt
ckpt = torch.load(CHECKPOINT)
model.load_state_dict(ckpt['state_dict'], strict=True)
model = model.cuda()


pil_to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

infer_size = 1024

def remove_background(img, target_color=255):
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
        img = img.flip(dims=[-3])
    else:
        raise TypeError
    
    img_norm = NORMALIZE(img)
    
    rh, rw = None, None
    if w >= h:
        rh = infer_size
        rw = int(w / h * infer_size)
    else:
        rw = infer_size
        rh = int(h / w * infer_size)
    rh = rh - rh % 64
    rw = rw - rw % 64

    input_tensor = F.interpolate(img_norm, size=(rh, rw), mode='bilinear')
    
    with torch.no_grad():
        _, _, pred_fusion = model(input_tensor)[:3]

    # output segment
    pred_fusion = F.interpolate(pred_fusion, size=(h, w), mode='bilinear')

    pred_fusion = pred_fusion[:, 0].data.cpu()
    img = img.data.cpu().numpy()
    
    pred_fusion_img = np.stack([pred_fusion.numpy()] * 3, axis=1)

    # df_describe = pd.DataFrame(pred_fusion.flatten())
    # print(df_describe.describe())

    base = np.ones(img.shape) * target_color / 255
    distance = img - base
    new_img = (base + distance * pred_fusion_img) * 255
    
    return new_img, pred_fusion
        
        

