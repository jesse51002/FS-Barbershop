import sys
sys.path.insert(0, './')
sys.path.insert(0, './models/HairClassifier/')
import math

import torch

from models.ModelBase import Model

from dataset import INFERENCE_TRANSFORM
from mobilenetv3 import mobilenet_v3_large
from train import NUM_CLASSES
from models.face_parsing.classes import CLASSES
from preprocess import BACKGROUND_COLOR




MODEL_PATH = "pretrained_models/hair_classifer_weights.pth"


class HairClassifier(Model):
    def __init__(self, weight_pth=MODEL_PATH, device="cuda"):
        self.name = "HairClassifer"
        
        self.model = mobilenet_v3_large(num_classes=NUM_CLASSES)
        self.model.load_state_dict(torch.load(weight_pth))
        self.model.eval()
        self.model = self.model.to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
    def predict(self, image, face_parse_mask):
        """
        Predicts the class of an image using a pre-trained model.

        Args:
            image (torch.Tensor): The input image tensor.
                Expected Shape = (1, 3, h, w)
                Expected Range = [0, 1]
            face_parse_mask (torch.Tensor): The face parsing mask tensor.
                Expected Shape: (1, h, w)

        Returns:
            torch.Tensor: The predicted class of the image.

        Description:
            This function takes an input image and a face parsing mask as input. It first crops the image using the face parsing mask using the `crop_img_from_mask` function. Then, it applies the `INFERENCE_TRANSFORM` to the cropped image. Finally, it moves the image to the device used by the pre-trained model and returns the predicted class of the image using the pre-trained model.
        """
        image = crop_img_from_mask(image, face_parse_mask)
        image = INFERENCE_TRANSFORM(image)
        image = image.to(self.model.device)
        return self.model(image)


def crop_img_from_mask(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Dont need the gradient for the mask, add uneeded extra time
    mask = mask.squeeze().detach()
    img = img.squeeze()
    
    img_mask = torch.stack([mask] * 3, dim=0)
    img = torch.where(img_mask == CLASSES["hair"], img, BACKGROUND_COLOR / 255)
    
    left, right, bottom, top = get_hair_box(mask)
    
    original_bounded_hair = img[:, bottom:top, left:right]
                
    # Turns the bounds into a box
    vertical_size = top - bottom
    horizontal_size = right - left
    max_size = max(vertical_size, horizontal_size)
    
    l_bounds, r_bounds, b_bounds, t_bounds = 0, 0, 0, 0
                
    if vertical_size > horizontal_size:
        diff = vertical_size - horizontal_size
        l_bounds = math.floor(diff / 2)
        r_bounds = math.ceil(diff / 2)
    elif vertical_size < horizontal_size:
        diff = horizontal_size - vertical_size
        b_bounds = math.floor(diff / 2)
        t_bounds = math.ceil(diff / 2)
    
    square_hair_img = torch.ones((3, original_bounded_hair.shape[-2], original_bounded_hair.shape[-1])).to(img.device) * (BACKGROUND_COLOR / 255)
    square_hair_img[:, b_bounds: max_size - t_bounds, l_bounds: max_size - r_bounds] = original_bounded_hair
    
    return square_hair_img


def get_hair_box(mask: torch.Tensor) -> tuple[int, int, int, int]:
    left, right, bottom, top = 0, 0, 0, 0
    # Gets positions where the data is not 0
    contains_pos = torch.argwhere(mask == CLASSES["hair"])
        
    # Handles empty case
    if contains_pos.shape[0] == 0:
        return right, left, bottom, top
        
    # min index where element is not zero
    mins = torch.min(contains_pos, axis=0).values
    bottom = mins[1]
    left = mins[2]
        
    # Max index where element is not zero
    maxs = torch.max(contains_pos, axis=0).values
    top = maxs[1]
    right = maxs[2]

    return left, right, bottom, top