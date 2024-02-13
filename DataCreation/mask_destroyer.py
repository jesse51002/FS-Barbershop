import os
import random
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

from mask_viewer import vis_seg
from PIL import Image, ImageDraw
from skimage import draw


DESTROYED_DATA_ROOT = "./MaskData/DestroyedDataTest"

MASK_ROOT = "./MaskData/ParsedData/masks"
IMAGE_ROOT = "./MaskData/ParsedData/images"


NO_DESTROY_PERC = 0.05
DESTROY_TYPES_OPTIONS = [
    ["hair", "circle", "irregular", "crop"],
    ["hair", "circle", "brush",  "crop"],
    ["hair", "circle"]
]

CROP_DIRECTIONS = ["top", "bottom", "right", "left"]


def create_multi_dim_mask(mask):
    """
    Input:
        mask (np.ndarray): 2d mask

    Output:
        mask (np.ndarray): mask with [h, w, 2] shape

    Creates a mask with 2 dimesions dimesion 0 with non hair, then dimesion 1 with hair.
    This improves masks fixing as occlusion doesnt cover face shape data.
    """
    
    # Remvoes third axis
    multi_dim_mask = np.zeros((mask.shape[0], mask.shape[1], 2))
        
    # makes all axis zeros second axis
    multi_dim_mask[:,:,0] = mask
    # Moves the hair mask to 2nd channel
    multi_dim_mask[multi_dim_mask[:,:,0] == 10, 1] = 10
    multi_dim_mask[multi_dim_mask[:,:,0] == 10, 0] = 0

    return multi_dim_mask

def destroy_mask(multi_dim_mask):    
    """
    Input:
        mask (np.ndarray): mask with [h, w, 2] shape

    Output:
        destroyed_mask (np.ndarray): mask with [h, w, 2] shape

    Adds holes and distrotions to the hair of the input mask for mask fixing training
    """
    combination = []

    dont_destroy = random.random() <= NO_DESTROY_PERC

    if not dont_destroy:
        combination = DESTROY_TYPES_OPTIONS[random.randint(0, len(DESTROY_TYPES_OPTIONS) - 1)]

    destroyed_mask = multi_dim_mask.copy()
    for destroy_type in combination:
        if destroy_type == "hair":
            destroyed_mask = hair_destroy(destroyed_mask)
        elif destroy_type == "brush": 
            destroyed_mask = brush_stroke_mask(destroyed_mask)
        elif destroy_type == "irregular":
            destroyed_mask = get_irregular_mask(destroyed_mask)
        elif destroy_type == "crop":
            destroyed_mask = crop_destroy(destroyed_mask)
        elif destroy_type == "circle":
            destroyed_mask = circles_destroy(destroyed_mask)

    return destroyed_mask
    
            
def test_function(test_amount=100, visualize=True):
    if not os.path.isdir(DESTROYED_DATA_ROOT):
        os.makedirs(DESTROYED_DATA_ROOT)
    else:
        for name in os.listdir(DESTROYED_DATA_ROOT):
            file_pth = os.path.join(DESTROYED_DATA_ROOT, name)
            if not os.path.isfile(file_pth):
                continue
            os.remove(file_pth)
    
    names = os.listdir(MASK_ROOT)
    random.shuffle(names)
    names = names[:min(test_amount, len(names))]

    for mask_name in names:
        print(mask_name)
        msk_pth = os.path.join(MASK_ROOT, mask_name)

        mask = cv2.imread(msk_pth)[:,:, 0]

        multi_dim_mask = create_multi_dim_mask(mask)
        destroyed_mask = destroy_mask(multi_dim_mask)

        if visualize:
            # create figure 
            fig = plt.figure(figsize=(10, 7)) 
                
            rows = 1
            columns = 2
                
            # Visualizes original
            fig.add_subplot(rows, columns, 1) 
            plt.imshow(vis_seg(get_vis_mask(multi_dim_mask))) 
            plt.axis('off') 
            plt.title("original") 
                    
            # Visualzies the destroyed masks
            fig.add_subplot(rows, columns, 2) 
            plt.imshow(vis_seg(get_vis_mask(destroyed_mask))) 
            plt.axis('off') 
            plt.title("Destroyed Mask") 
                
            
            # plt.show()
            plt.savefig(os.path.join(DESTROYED_DATA_ROOT, mask_name))
            plt.close()

def get_vis_mask(mask):
    vis_mask = np.where(mask[:,:, 1] != 0, mask[:,:, 1], mask[:,:, 0])
    return vis_mask

# Calculates the bounding box information from the semantic label
def calculate_bound_box(mask):
    top, bottom, right, left = 0,0,0,0 
    
    # Gets positions where the data is not 0
    contains_pos = np.argwhere(mask != 0)
    
    # Handles empty case
    if contains_pos.shape[0] == 0:
        return right, left, bottom, top
    
    # min index where element is not zero
    mins = np.min(contains_pos, axis = 0)

    bottom = mins[0]
    right = mins[1]
    
    # Max index where element is not zero
    maxs = np.max(contains_pos, axis = 0)
    top = maxs[0]
    left = maxs[1]
    
    return right, left, bottom, top

def crop_destroy(mask: np.ndarray, amount=None):    
    crop_dest_mask = mask.copy()
    
    right, left, bottom, top  = calculate_bound_box(crop_dest_mask)
    mid_x, mid_y = int((right + left) / 2), int((bottom + top) / 2)

    if amount is None:
        amount = random.randint(1, len(CROP_DIRECTIONS))
    
    direction_options = CROP_DIRECTIONS.copy()
    
    # Amount that random float is squared to increase the chances of lower numbers
    lower_priority = 4
    
    for _ in range(amount):
        rand_idx = random.randint(0, len(direction_options) - 1)
        
        chosen_dir = direction_options[rand_idx]
        direction_options.pop(rand_idx)
        
        destroy_amount = pow(random.random(), lower_priority)
                
        if chosen_dir == "top":
            start_idx = int(top - destroy_amount * (top - mid_y))
            crop_dest_mask[start_idx:] = 0
        elif chosen_dir == "bottom":
            start_idx = int(bottom + destroy_amount * (top - mid_y))
            crop_dest_mask[:start_idx] = 0
        elif chosen_dir == "left":
            start_idx = int(left - destroy_amount * (left - mid_x))
            crop_dest_mask[:, start_idx:] = 0
        elif chosen_dir == "right":
            start_idx = int(right + destroy_amount * (left - mid_x))
            crop_dest_mask[:, :start_idx] = 0
    
    return crop_dest_mask

def hair_destroy(mask: np.ndarray, amount=None):
    # All category bounding box
    right, left, bottom, top  = calculate_bound_box(mask)
    
    
    max_move_perc = 0.05
    max_scale_perc = 0.25
    
    # hair bounding box
    hair_right, hair_left, hair_bottom, hair_top = calculate_bound_box(np.where(mask == 10, 1, 0))
    
    # gets hair mask
    unscaled_hair_mask = np.where(mask[hair_bottom:hair_top, hair_right:hair_left, 1] == 10, 1, 0)

    # hair scaling
    scale_amount = 1 + ((max_scale_perc * 2 * random.random()) - max_scale_perc)
    current_x_size = hair_left - hair_right
    new_x_size = int(scale_amount * current_x_size)

    # Moves the mask to match the scaling
    hair_right -= math.ceil((new_x_size - current_x_size) / 2)
    hair_left += math.floor((new_x_size - current_x_size) / 2)
    
    scaled_hair_mask = cv2.resize(unscaled_hair_mask, (new_x_size, hair_top - hair_bottom), interpolation=cv2.INTER_NEAREST)

    
    # Amount to move the hair, this works for both positive and negative movement
    hair_move_x = int((left - right) * 2 * random.random() * max_move_perc - ((left - right) * max_move_perc)) 
    hair_move_y = int((top - bottom) * 2 * random.random() * max_move_perc - ((top - bottom) * max_move_perc)) 
    
    if hair_right + hair_move_x < 0:
        scaled_hair_mask = scaled_hair_mask[:, -(hair_right + hair_move_x):]
        hair_right += -(hair_right + hair_move_x)
    if hair_bottom + hair_move_y < 0:
        scaled_hair_mask = scaled_hair_mask[-(hair_bottom + hair_move_y):]
        hair_bottom += -(hair_bottom + hair_move_y)
    
    if hair_left + hair_move_x >= mask.shape[1]:
        scaled_hair_mask = scaled_hair_mask[:, : -(1 + (hair_left + hair_move_x) - mask.shape[1])]
        hair_left -= 1 + (hair_left + hair_move_x) - mask.shape[1]
    if hair_top + hair_move_y >= mask.shape[0]:
        scaled_hair_mask = scaled_hair_mask[: -(1 + (hair_top + hair_move_y) - mask.shape[0])]
        hair_top -= 1 + (hair_top + hair_move_y) - mask.shape[0]
    
    
    
    # Creates a moved mask for the just the hair
    moved_hair_mask_scaled = np.zeros_like(mask[:,:, 1])
    moved_hair_mask_scaled[
        hair_bottom + hair_move_y : hair_top + hair_move_y,
        hair_right + hair_move_x : hair_left + hair_move_x
        ] = scaled_hair_mask
    
    mask_hair_moved = mask.copy()
    
    # Shifts the current hair
    mask_hair_moved[:,:, 1] = np.where(moved_hair_mask_scaled == 1, 10, 0)
    
    return mask_hair_moved

def brush_stroke_mask(img,
                      num_vertices=(4, 12),
                      mean_angle=2 * math.pi / 5,
                      angle_range=2 * math.pi / 15,
                      brush_width=(5, 40),
                      max_loops=4,
                      dtype='uint8'):
    """Generate free-form mask.

    The method of generating free-form mask is in the following paper:
    Free-Form Image Inpainting with Gated Convolution.

    When you set the config of this type of mask. You may note the usage of
    `np.random.randint` and the range of `np.random.randint` is [left, right).

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    TODO: Rewrite the implementation of this function.

    Args:
        img_shape (tuple[int]): Size of the image.
        num_vertices (int | tuple[int]): Min and max number of vertices. If
            only give an integer, we will fix the number of vertices.
            Default: (4, 12).
        mean_angle (float): Mean value of the angle in each vertex. The angle
            is measured in radians. Default: 2 * math.pi / 5.
        angle_range (float): Range of the random angle.
            Default: 2 * math.pi / 15.
        brush_width (int | tuple[int]): (min_width, max_width). If only give
            an integer, we will fix the width of brush. Default: (12, 40).
        max_loops (int): The max number of for loops of drawing strokes.
        dtype (str): Indicate the data type of returned masks.
            Default: 'uint8'.

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    img_h, img_w = img.shape[:2]
    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        raise TypeError('The type of num_vertices should be int'
                        f'or tuple[int], but got type: {num_vertices}')

    if isinstance(brush_width, tuple):
        min_width, max_width = brush_width
    elif isinstance(brush_width, int):
        min_width, max_width = brush_width, brush_width + 1
    else:
        raise TypeError('The type of brush_width should be int'
                        f'or tuple[int], but got type: {brush_width}')

    average_radius = math.sqrt(img_h * img_h + img_w * img_w) / 8
    mask = Image.new('L', (img_w, img_h), 0)

    loop_num = np.random.randint(1, max_loops)
    num_vertex_list = np.random.randint(
        min_num_vertices, max_num_vertices, size=loop_num)
    angle_min_list = np.random.uniform(0, angle_range, size=loop_num)
    angle_max_list = np.random.uniform(0, angle_range, size=loop_num)

    for loop_n in range(loop_num):
        num_vertex = num_vertex_list[loop_n]
        angle_min = mean_angle - angle_min_list[loop_n]
        angle_max = mean_angle + angle_max_list[loop_n]
        angles = []
        vertex = []

        # set random angle on each vertex
        angles = np.random.uniform(angle_min, angle_max, size=num_vertex)
        reverse_mask = (np.arange(num_vertex, dtype=np.float32) % 2) == 0
        angles[reverse_mask] = 2 * math.pi - angles[reverse_mask]

        h, w = mask.size

        # set random vertices
        vertex.append((np.random.randint(0, w), np.random.randint(0, h)))
        r_list = np.random.normal(
            loc=average_radius, scale=average_radius // 2, size=num_vertex)
        for i in range(num_vertex):
            r = np.clip(r_list[i], 0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))
        # draw brush strokes according to the vertex and angle list
        draw = ImageDraw.Draw(mask)
        width = np.random.randint(min_width, max_width)
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2, v[1] - width // 2,
                          v[0] + width // 2, v[1] + width // 2),
                         fill=1)
    # randomly flip the mask
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.array(mask).astype(dtype=getattr(np, dtype))
    mask = mask[:, :]
    
    new_img = img.copy()
    new_img[mask == 1, :] = 0
    return new_img


def random_irregular_mask(img,
                          num_vertices=(4, 8),
                          max_angle=4,
                          length_range=(10, 100),
                          brush_width=(5, 40),
                          dtype='uint8'):
    """Generate random irregular masks.

    This is a modified version of free-form mask implemented in
    'brush_stroke_mask'.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    TODO: Rewrite the implementation of this function.

    Args:
        img_shape (tuple[int]): Size of the image.
        num_vertices (int | tuple[int]): Min and max number of vertices. If
            only give an integer, we will fix the number of vertices.
            Default: (4, 8).
        max_angle (float): Max value of angle at each vertex. Default 4.0.
        length_range (int | tuple[int]): (min_length, max_length). If only give
            an integer, we will fix the length of brush. Default: (10, 100).
        brush_width (int | tuple[int]): (min_width, max_width). If only give
            an integer, we will fix the width of brush. Default: (10, 40).
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=dtype)
    if isinstance(length_range, int):
        min_length, max_length = length_range, length_range + 1
    elif isinstance(length_range, tuple):
        min_length, max_length = length_range
    else:
        raise TypeError('The type of length_range should be int'
                        f'or tuple[int], but got type: {length_range}')
    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        raise TypeError('The type of num_vertices should be int'
                        f'or tuple[int], but got type: {num_vertices}')

    if isinstance(brush_width, int):
        min_brush_width, max_brush_width = brush_width, brush_width + 1
    elif isinstance(brush_width, tuple):
        min_brush_width, max_brush_width = brush_width
    else:
        raise TypeError('The type of brush_width should be int'
                        f'or tuple[int], but got type: {brush_width}')

    num_v = np.random.randint(min_num_vertices, max_num_vertices)

    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        # from the start point, randomly setlect n \in [1, 6] directions.
        direction_num = np.random.randint(1, 6)
        angle_list = np.random.randint(0, max_angle, size=direction_num)
        length_list = np.random.randint(
            min_length, max_length, size=direction_num)
        brush_width_list = np.random.randint(
            min_brush_width, max_brush_width, size=direction_num)
        for direct_n in range(direction_num):
            angle = 0.01 + angle_list[direct_n]
            if i % 2 == 0:
                angle = 2 * math.pi - angle
            length = length_list[direct_n]
            brush_w = brush_width_list[direct_n]
            # compute end point according to the random angle
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1, brush_w)
            start_x, start_y = end_x, end_y
    mask = np.expand_dims(mask, axis=2)

    return mask[:, :, 0]


def get_irregular_mask(img, area_ratio_range=(0.05, 0.5), **kwargs):
    """Get irregular mask with the constraints in mask ratio

    Args:
        img_shape (tuple[int]): Size of the image.
        area_ratio_range (tuple(float)): Contain the minimum and maximum area
        ratio. Default: (0.15, 0.5).

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    mask = random_irregular_mask(img, **kwargs)
    min_ratio, max_ratio = area_ratio_range

    while not min_ratio < (np.sum(mask) /
                           (img.shape[0] * img.shape[1])) < max_ratio:
        mask = random_irregular_mask(img, **kwargs)

    new_img = img.copy()
    new_img[mask == 1, :] = 0
    return new_img

def circles_destroy(mask: np.ndarray, min_circles=2, max_circles=5):    
    cir_dest_mask = mask.copy()
    
    right, left, bottom, top  = calculate_bound_box(cir_dest_mask)

    amount = random.randint(min_circles, max_circles)

    min_perc = 0.05
    max_perc = 0.15

    # Amount that random float is squared to increase the chances of lower numbers
    lower_priority = 3


    for _ in range(amount):
        # Calculates the random center
        center_y = int(bottom + (top - bottom) * random.random())
        center_x = int(right + (left - right) * random.random())

        # Uses the larger bound direction as a scaling refernce
        square_size = max((top - bottom), (left - right))

        # Gets the scaled size of the x and y axis ---

        rand_y_max_size = square_size * (min_perc + (max_perc - min_perc) * random.random())
        relative_y_size = min_perc + (max_perc - min_perc) * math.pow(random.random(), lower_priority)
        size_y = int(relative_y_size * (rand_y_max_size * 0.75) + (rand_y_max_size * 0.25))

        rand_x_max_size = square_size * (min_perc + (max_perc - min_perc) * random.random())
        relative_x_size = min_perc + (max_perc - min_perc) * math.pow(random.random(), lower_priority)
        size_x = int(relative_x_size * (rand_x_max_size * 0.75) + (rand_x_max_size * 0.25))

        # ---

        # Gets a ranodm angle
        rand_angle = 2 * math.pi * random.random()

        # Calculatse ellipse mask and sets it to zero
        rr, cc = draw.ellipse(center_y, center_x, size_y, size_x, shape=cir_dest_mask.shape, rotation=rand_angle)

        cir_dest_mask[rr, cc] = 0


    return cir_dest_mask

if __name__ == "__main__":
    test_function()




