
import numpy as np
import os
import PIL
from PIL import Image
from models.face_parsing.classes import CLASSES
from skimage.morphology import binary_dilation


DILATION_AMOUNT = 3


def expand_face_mask(face_seg, human_seg):
    # Dilates the hair to catch the small silver of body segs because otherwise it will outline the hair
    hair_mask = np.where(face_seg == CLASSES["hair"], 1, 0)
    hair_mask = binary_dilation(hair_mask, np.ones((5, 5)))
    face_seg = np.where((hair_mask == 1) & (human_seg == 1) & (face_seg == 0), CLASSES["hair"], face_seg)

    # --------- Adds body to the face parsing
    body_mask = np.zeros(human_seg.shape[-2:])
        
    # Only keeps the body mask below the nose
    # The face segmentation model will handle the face area
    nose_idx = np.where(face_seg == CLASSES["nose"])
    max_nose = int(human_seg.shape[-1] / 2)
    if nose_idx[0].shape[0] != 0:
        max_nose = nose_idx[0].max()
        
    human_mask_og = np.where(human_seg > 0.5, np.ones_like(human_seg), np.zeros_like(human_seg))
    human_mask_og[:max_nose] = 0
    body_mask[human_mask_og == 1] = 1

    # Removes places that used to be hair in mask1
    body_mask[(human_seg == 0) | (face_seg != 0)] = 0

    # Adds body mask
    face_seg = np.where(body_mask == 1, CLASSES["body"], face_seg)

    # Adds uncaught strands on hair
    hair_mask = np.where(face_seg == CLASSES["hair"], 1, 0)
    for i in range(DILATION_AMOUNT):
        hair_mask = binary_dilation(hair_mask, np.ones((10, 10)))
    face_seg = np.where((hair_mask == 1) & (human_seg == 1) & (face_seg == 0), CLASSES["hair"], face_seg) 
    
    # ---------

    # removes over segmented parts
    face_seg = np.where(human_seg < 0.1, 0, face_seg)

    return face_seg
    

def vis_seg(pred):
    num_labels = 16

    color = np.array([[0, 0, 0],  ## 0
                      [102, 204, 255],  ## 1
                      [255, 204, 255],  ## 2
                      [255, 255, 153],  ## 3
                      [255, 153, 153],  ## 4
                      [255, 255, 102],  ## 5
                      [51, 255, 51],  ## 6
                      [0, 153, 255],  ## 7
                      [0, 255, 255],  ## 8
                      [0, 0, 255],  ## 9
                      [204, 102, 255],  ## 10
                      [0, 153, 255],  ## 11
                      [0, 255, 153],  ## 12
                      [0, 51, 0],  # 13
                      [102, 153, 255],  ## 14
                      [255, 153, 102],  ## 15
                      [255, 255, 0],  ## 16
                      [255, 0, 255],  ## 17
                      [255, 255, 255],  ## 18
                      ])
    h, w = np.shape(pred)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     print(color.shape)
    for ii in range(num_labels):
        #         print(ii)
        mask = pred == ii
        rgb[mask, None] = color[ii, :]
    # Correct unk
    unk = pred == 255
    rgb[unk, None] = color[0, :]
    return rgb


def save_vis_mask(img_path1, img_path2, sign, output_dir, mask):
    im_name_1 = os.path.splitext(os.path.basename(img_path1))[0]
    im_name_2 = os.path.splitext(os.path.basename(img_path2))[0]
    vis_path = os.path.join(output_dir, 'vis_mask_{}_{}_{}.png'.format(im_name_1, im_name_2, sign))
    vis_mask = vis_seg(mask)
    PIL.Image.fromarray(vis_mask).save(vis_path)

def save_original_mask(img_path, output_dir, mask, return_numpy=False):
    im_name = os.path.splitext(os.path.basename(img_path))[0]
    vis_path = os.path.join(output_dir, 'org_mask_{}.png'.format(im_name))
    vis_mask = vis_seg(mask)
    if return_numpy:
        return vis_mask
    PIL.Image.fromarray(vis_mask).save(vis_path)

def save_human_mask(img_path1, img_path2, sign, output_dir, mask):
    im_name_1 = os.path.splitext(os.path.basename(img_path1))[0]
    im_name_2 = os.path.splitext(os.path.basename(img_path2))[0]
    vis_path = os.path.join(output_dir, 'vis_mask_human_{}_{}_{}.png'.format(im_name_1, im_name_2, sign))
    vis_mask = vis_seg(mask)
    PIL.Image.fromarray(vis_mask).save(vis_path)

