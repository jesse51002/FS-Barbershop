import sys
sys.path.insert(0,'./src')
sys.path.insert(0,'./src/HumanParsing')

# This error  subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1.
# This has the fix https://github.com/HRNet/HRNet-Semantic-Segmentation/issues/39
# Then restart your vscode


import time
import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils import data
from tqdm import tqdm
from PIL import Image as PILImage
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import Constants

import networks
from hp_utils.transforms import BGR2RGB_transform

from Utils import vis_seg
from Model import Model


MODEL_PATH = './src/HumanParsing/checkpoints/exp-schp-201908261155-lip.pth'

class body_model(Model):
    
    def __init__(self):
        self.max_size = 1024
        self.model = networks.init_model('resnet101', num_classes=20, pretrained=None)
        
        self.IMAGE_MEAN = self.model.mean
        self.IMAGE_STD = self.model.std
        self.INPUT_SPACE = self.model.input_space
        print('image mean: {}'.format(self.IMAGE_MEAN))
        print('image std: {}'.format(self.IMAGE_STD))
        print('input space:{}'.format(self.INPUT_SPACE))
        
        if self.INPUT_SPACE == 'BGR':
            self.transform = transforms.Compose([
                transforms.Normalize(mean=self.IMAGE_MEAN,
                                    std=self.IMAGE_STD),

            ])
        if self.INPUT_SPACE == 'RGB':
            print('RGB Transformation')
            self.transform = transforms.Compose([
                BGR2RGB_transform(),
                transforms.Normalize(mean=self.IMAGE_MEAN,
                                    std=self.IMAGE_STD),
            ])
        
        # Load model weight
        state_dict = torch.load(MODEL_PATH)['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.cuda()
        self.model.eval()
        
    @torch.no_grad()
    def inference(self, img):
        h,w = None, None
        if isinstance(img, str):
            img = cv2.imread(img)
            h, w = img.shape[:2]
            img = torch.tensor(img).permute((2,0,1)).unsqueeze(0).float()
        elif isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()
        elif isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:
            raise TypeError

        max_shape = max(h, w)
        if max_shape > self.max_size:
            scale_reduce = self.max_size / max_shape
            
            downsample = torch.nn.Upsample(scale_factor=scale_reduce, mode='bilinear', align_corners=True)
            
            img = downsample(img)
        
        img = self.transform(img / 255).cuda()
        
        parsing_output = self.model(img)
        parsing_output = parsing_output[0][-1].detach().cpu()

        scale_num = 1024 / max(parsing_output.shape[2:])
        
        interp = torch.nn.Upsample(scale_factor=scale_num, mode='bilinear', align_corners=True)

        output_perc = interp(parsing_output)
        output = output_perc.argmax(dim=1)
        
        # output[output_perc[:,0] >= BACKGROUND_PRIORITY] = 0
        
        return output, output_perc
    
    
if __name__ == "__main__":
    face_dir = os.path.join(Constants.RAW_BODY_IMAGES_DIR, 'test')
    mask_dir = os.path.join(Constants.RAW_BODY_IMAGES_DIR, 'masks')
    
    accepted_imgs = ["png", "jpg", "jpeg"]
    
    model = body_model()
    
    torch.no_grad()
    for img_name in os.listdir(face_dir):
        if img_name.split(".")[-1] not in accepted_imgs:
            continue

        print(img_name)
        img_pth = os.path.join(face_dir, img_name)
        img = cv2.imread(img_pth)
        print("Img shape:",img.shape)

        start = time.time()
        output_img, output_perc = model.inference(img)
        output_img = output_img.detach().cpu().numpy()[0]
        print("Took:", time.time() - start)
        print("output:", output_img.shape)
        
        # setting values to rows and column variables 
        rows = 1
        columns = 2
        
        # create figure 
        fig = plt.figure(figsize=(10, 7)) 
        # Adds a subplot at the 1st position 
        fig.add_subplot(1, 2, 1) 
        
        # showing image 
        plt.imshow(img[:,:,::-1]) 
        plt.axis('off') 
        plt.title("Image") 
        
        # Adds a subplot at the 2nd position 
        fig.add_subplot(rows, columns, 2) 
        
        # showing image 
        plt.imshow(vis_seg(output_img))
        plt.axis('off') 
        plt.title("mask") 
        
        plt.savefig(os.path.join(mask_dir, img_name))
        # plt.show()
        