import os
import PIL
import torchvision
import torch.nn.functional as F
from PIL import Image
import torch
import torchvision
from PIL import Image
from utils.bicubic import BicubicDownSample
import numpy as np
from models.FacerParsing import facer_to_bisnet

from torchvision.transforms import transforms
import scipy


def load_image(img_path, normalize=True, downsample=False):
    img = PIL.Image.open(img_path).convert('RGB')
    if downsample:
        img = img.resize((256, 256), PIL.Image.LANCZOS)
    img = transforms.ToTensor()(img)
    if normalize:
        img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
    return img


def dilate_erosion_mask_path(opts, im_path, dilate_erosion=5):
    # # Mask
    # mask = Image.open(mask_path).convert("RGB")
    # mask = mask.resize((256, 256), PIL.Image.NEAREST)
    # mask = transforms.ToTensor()(mask)  # [0, 1]

    mask_save_dir = os.path.join(opts.output_dir, 'masks')
    im_name = os.path.basename(im_path).split(".")[0]
    mask_npy_pth = os.path.join(mask_save_dir, im_name) + ".npy"
        
    mask = torch.tensor(np.load(mask_npy_pth)).float().unsqueeze(0)
    mask = torch.where(mask == 10, torch.ones_like(mask), torch.zeros_like(mask))
    mask = F.interpolate(mask.unsqueeze(0), size=(256, 256), mode='nearest').squeeze()

    # Hair mask + Hair image
    hair_mask = mask
    hair_mask = hair_mask.numpy()
    hair_mask_dilate = scipy.ndimage.binary_dilation(hair_mask, iterations=dilate_erosion)
    hair_mask_erode = scipy.ndimage.binary_erosion(hair_mask, iterations=dilate_erosion)

    hair_mask_dilate = np.expand_dims(hair_mask_dilate, axis=0)
    hair_mask_erode = np.expand_dims(hair_mask_erode, axis=0)

    return torch.from_numpy(hair_mask_dilate).float(), torch.from_numpy(hair_mask_erode).float()


def dilate_erosion_mask_tensor(mask, dilate_erosion=5):
    hair_mask = mask.clone()
    hair_mask = hair_mask.numpy()
    hair_mask_dilate = scipy.ndimage.binary_dilation(hair_mask, iterations=dilate_erosion)
    hair_mask_erode = scipy.ndimage.binary_erosion(hair_mask, iterations=dilate_erosion)

    hair_mask_dilate = np.expand_dims(hair_mask_dilate, axis=0)
    hair_mask_erode = np.expand_dims(hair_mask_erode, axis=0)

    return torch.from_numpy(hair_mask_dilate).float(), torch.from_numpy(hair_mask_erode).float()
