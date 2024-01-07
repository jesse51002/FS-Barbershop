import sys
sys.path.insert(0,'./')
sys.path.insert(0,'./human_parse')

import os
import argparse
import numpy as np
import torch

import cv2
import torch
import torchvision.transforms as transforms
import networks
from utils.seg_utils import save_original_mask

CHECKPOINT = "./human_parse/exp-schp-201908270938-pascal-person-part.pth"

H, W = 1024, 1024
upsample = torch.nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)


model = networks.init_model('resnet101', num_classes=7, pretrained=None)
# Load model weight
state_dict = torch.load(CHECKPOINT)['state_dict']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.cuda()
model.eval()

IMAGE_MEAN = model.mean
IMAGE_STD = model.std
INPUT_SPACE = model.input_space
print(f"Input space: {INPUT_SPACE}")


normalize = transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
])

def get_segmentation(img):
    
    if isinstance(img, str):
        img = cv2.imread(img)
        img = transform(img / 255).float().unsqueeze(0).cuda()
    elif isinstance(img, np.ndarray):
        img = transform(img / 255).float().unsqueeze(0).cuda()
    elif isinstance(img, torch.Tensor):
        img = normalize(img).flip(dims=[1])
    else:
        raise TypeError

    og_h, og_w = img.shape[-2:]
    scaled_im = upsample(img)

    parsing_output = model(scaled_im)
    parsing_output = parsing_output[0][-1]
    output = torch.nn.Upsample(size=(og_h, og_w), mode='bilinear', align_corners=True)(parsing_output)
    return output


if __name__ == "__main__":
    face_dir = "./input/face/"
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
    
        seg = get_segmentation(img).argmax(1).detach().cpu().numpy()[0]
        save_original_mask(img_pth, "./output/masks", seg)
