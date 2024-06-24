import sys
sys.path.insert(0, './')
from PIL import Image
import cv2
import torch
import clip
import os
from enum import Enum

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms import InterpolationMode
from models.ModelBase import Model


class HairColorEnum(Enum):
    BLACK = 0
    BRUNNETE = 1
    BLONDE = 2
    RED = 3
    BLUE = 4
    ORANGE = 5
    GREEN = 6
    WHITE = 7
    PURPLE = 8
    YELLOW = 9
    PINK = 10
    SILVER = 11


class QualityClip(Model):
    def __init__(self, device="cuda"):
        self.name = "QualityClip"
        
        # load model
        self.weights = "pretrained_models/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"
        
        self.model, self.preprocess = clip.load("ViT-B/16", device="cpu")
        self.model = self.model.cuda()
        # farl_state = torch.load(self.weights)
        # self.model.load_state_dict(farl_state["state_dict"], strict=False)

        self.quality_options = [
             "HD, canon, 8k, high quality, photo",
            "water color, painting, blurry, greyscale, waterpainting, black and white, low quality"
        ]
        
        """
        self.hair_colors = [
            "a person with black hair",
            "a person with brunnete hair",
            "a person with blonde hair",
        ]
        """
        
        # Quality tokenizer
        self.quality_text = clip.tokenize(self.quality_options).to(device)
        self.model.encode_text(self.quality_text)

        self.transforms = Compose([
            Resize(self.model.visual.input_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(self.model.visual.input_resolution),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
           
    def inference(self, image):
        self.model.encode_image(image)
        logits_per_image, logits_per_text = self.model(image, self.quality_text)
        quality_probs = logits_per_image.softmax(dim=-1)
        return quality_probs

    def img_preprocess(self, img):
        return self.preprocess(Image.fromarray(img)).unsqueeze(0).cuda()
    
    def torch_prepreocess(self, img):
        return self.transforms(torch.flip(img, dims=[1]))


if __name__ == "__main__":
    model = QualityClip()
    
    input_dir = "./input/face"
    list_dir = os.listdir(input_dir)
    list_dir.sort()
    for x in list_dir:
        if x == ".ipynb_checkpoints":
            continue
        img_pth = os.path.join(input_dir, x)
        model_in = model.img_preprocess(cv2.imread(img_pth))

        inference = model.inference(model_in)
        model_percs = inference.argmax(1).data.cpu().numpy()
        # print(inference)
        print(os.path.basename(img_pth), model.hair_colors[model_percs[0]])