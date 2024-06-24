import torch
from torch import nn
from torchvision.io import read_image, ImageReadMode
import numpy as np
import os
from functools import partial
from utils.bicubic import BicubicDownSample
from tqdm import tqdm
import torchvision
from utils.data_utils import convert_npy_code
from models.face_parsing.model import seg_mean, seg_std
from models.face_parsing.classes import CLASSES
from losses.align_loss import AlignLossBuilder
import torch.nn.functional as F
import cv2
from utils.data_utils import load_FS_latent
from utils.seg_utils import save_vis_mask
from utils.data_utils import cuda_unsqueeze
from utils.image_utils import dilate_erosion_mask_tensor
from threading import Thread, Lock
from models.ModelBase import Model
import time

toPIL = torchvision.transforms.ToPILImage()
COLOR_MATCH_QUARTILE = 0.75


class Alignment(nn.Module):

    def __init__(
        self, opts: dict,
        net0: Model = None, net1: Model = None,
        seg0: Model = None, seg1: Model = None,
        quality_clip: Model = None,
        hair_classifier: Model = None
    ):
        """
        Initializes an instance of the Alignment class.
        Args:
            opts (object): An object containing options for the alignment.
            net0 (object): The first gpu neural network.
            net1 (object, optional): The second gpu neural network. Defaults to None.
            seg0 (object): The first gpu segmentation.
            seg1 (object, optional): The second gpu segmentation. Defaults to None.
        Returns:
            None
            
        Description:
            This function initializes an instance of the Alignment class.
            It sets the options, neural networks, and segmentations.
            It also sets the mean and standard deviation of the segmentations based on the device specified in the options. 
            If the options specify multi-GPU, it sets the mean and standard deviation of the second segmentation based on the second device. It then loads the downsampling and sets up the alignment loss builder.
        """        
        
        super(Alignment, self).__init__()
        
        self.opts = opts
        self.net0 = net0
        self.net1 = net1
        self.seg0 = seg0
        self.seg1 = seg1
        self.seg0.user_seg_mean = seg_mean.to(self.opts.device[0])
        self.seg0.user_seg_std = seg_std.to(self.opts.device[0])

        if self.opts.is_multi_gpu:
            self.seg1.user_seg_std = seg_std.to(self.opts.device[1])
            self.seg1.user_seg_mean = seg_mean.to(self.opts.device[1])

        self.quality_clip = quality_clip
        self.hair_classifier = hair_classifier

        self.load_downsampling()
        self.setup_align_loss_builder()

    def set_opts(self, opts: dict) -> None:
        self.opts = opts

    def load_downsampling(self) -> None:
        self.downsample = BicubicDownSample(factor=self.opts.size // 512)
        self.downsample_256 = BicubicDownSample(factor=self.opts.size // 256)

    def setup_align_loss_builder(self) -> None:
        self.loss_builder0 = AlignLossBuilder(self.opts, num_classes=len(CLASSES), device=self.opts.device[0])
        if self.opts.is_multi_gpu:
            self.loss_builder1 = AlignLossBuilder(self.opts, num_classes=len(CLASSES), device=self.opts.device[1])
            
    def create_target_segmentation_mask(self, img_path1: str, img_path2: str, sign: str, save_intermediate: bool=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Creates a target segmentation mask for two input images.

        Args:
            img_path1 (str): The path to the first input image.
            img_path2 (str): The path to the second input image.
            sign (int): The sign indicating the direction of the hair interpolation.
            save_intermediate (bool, optional): Whether to save intermediate visualizations. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the target mask, the hair mask for the first image, the hair mask for the second image, and the updated hair mask for the second image.
            
        Description:
            This function creates a target segmentation mask for two input images.
            It first loads the masks for the first and second images.
            It keeps the face from image 1 and the hair from image 2.
        """
        
        
        device = self.opts.device[0]

        mask_save_dir = os.path.join(self.opts.output_dir, 'masks')
        
        im1_name = os.path.basename(img_path1).split(".")[0]
        im2_name = os.path.basename(img_path2).split(".")[0]
        
        mask1_npz_pth = os.path.join(mask_save_dir, im1_name) + "_mask.npz"
        mask2_npz_pth = os.path.join(mask_save_dir, im2_name) + "_mask.npz"

        mask1_npz = np.load(mask1_npz_pth)
        mask2_npz = np.load(mask2_npz_pth)
        
        seg_target1 = torch.tensor(mask1_npz["mask"]).to(device).float().unsqueeze(0)
        seg_target2 = torch.tensor(mask2_npz["mask"]).to(device).float().unsqueeze(0)

        ggg = torch.where(seg_target1 == 0, torch.zeros_like(seg_target1), torch.ones_like(seg_target1))

        hair_mask1 = torch.where(seg_target1 == CLASSES["hair"], torch.ones_like(seg_target1), torch.zeros_like(seg_target1))
        seg_target1 = seg_target1[0].byte().cpu().detach()
        seg_target1 = torch.where(seg_target1 == CLASSES["hair"], torch.zeros_like(seg_target1), seg_target1)
        
        hair_mask2 = torch.where(seg_target2 == CLASSES["hair"], torch.ones_like(seg_target2), torch.zeros_like(seg_target2))
   
        # Updates hair mask with relatove jaw and brow interpolation points
        hair_mask2 = self.hair_interpolate_mask_updater(hair_mask2, mask1_npz, mask2_npz)
        
        # Adds the hair to the mask
        seg_target2 = torch.where((hair_mask2 == 0) & (seg_target2 == CLASSES["hair"]), 0, seg_target2)
        seg_target2 = torch.where(hair_mask2 == 1, CLASSES["hair"], seg_target2)
        hair_mask2 = torch.where(seg_target2 == CLASSES["hair"], torch.ones_like(seg_target2), torch.zeros_like(seg_target2))

        ggg = torch.where(seg_target2 == CLASSES["hair"], torch.ones_like(seg_target2), ggg).long()
        seg_target2 = seg_target2[0].byte().cpu().detach()
        
        OB_region = torch.where(
            (seg_target2 != CLASSES["hair"]) * (seg_target2 != CLASSES["background"]) * (seg_target2 != 15) * (
                    seg_target1 == 0),
            255 * torch.ones_like(seg_target1), torch.zeros_like(seg_target1))

        new_target = torch.where(seg_target2 == CLASSES["hair"], CLASSES["hair"] * torch.ones_like(seg_target1), seg_target1)

        inpainting_region = torch.where((new_target != CLASSES["background"]) * (new_target != CLASSES["hair"]), 255 * torch.ones_like(new_target),
                                        OB_region).numpy()
        tmp = torch.where(torch.from_numpy(inpainting_region) == 255, torch.zeros_like(new_target), new_target) / CLASSES["hair"]
        new_target_inpainted = (
                    cv2.inpaint(tmp.clone().numpy(), inpainting_region, 3, cv2.INPAINT_NS).astype(np.uint8) * CLASSES["hair"])
        new_target_final = torch.where(OB_region, torch.from_numpy(new_target_inpainted), new_target)
        target_mask = new_target_final.unsqueeze(0).long().cuda()

        # Adds hair between neck and hair
        # This is for situations where the hair doesnt connect to the neck because the previous nech was to wide
        left, right, bottom, top = self.get_hair_box(target_mask)
        
        # Crops the box inwards to not mess up the curve on the outer hair curve
        top_crop_perc = 0.2
        side_crop_perc = 0.5

        top_move_amount = int(top_crop_perc * (top - bottom))
        side_move_amount = int(side_crop_perc / 2 * (right - left))

        left += side_move_amount
        right -= side_move_amount
        bottom = min(bottom + 70, top_move_amount + bottom)

        binary_box_mask = torch.zeros_like(target_mask)
        binary_box_mask[:, bottom:top, right:left] = 1

        target_mask = torch.where((binary_box_mask == 1) & (target_mask == 0), CLASSES["hair"], target_mask)
        
        # Visualize the hair
        hair_mask_target = torch.where(target_mask == CLASSES["hair"], torch.ones_like(target_mask), torch.zeros_like(target_mask))
        hair_mask_target = hair_mask_target.float().unsqueeze(0)
        
        # ---------------  Save Visualization of Target Segmentation Mask
        masks_dir = os.path.join(self.opts.output_dir, "masks")
        if save_intermediate:
            save_vis_mask(img_path1, img_path2, sign, masks_dir, target_mask.squeeze().cpu())

        return target_mask, hair_mask_target, hair_mask1, hair_mask2

    def hair_interpolate_mask_updater(self, hair_mask2: torch.Tensor, mask1_npz: dict, mask2_npz: dict) -> torch.Tensor:
        """
        Interpolates the hair mask based on the provided hair mask, mask1_npz, and mask2_npz.
        Parameters:
            hair_mask2 (torch.Tensor): The original hair mask.
            mask1_npz (dict): The dictionary containing the keypoints of mask1.
            mask2_npz (dict): The dictionary containing the keypoints of mask2.
        Returns:
            torch.Tensor: The interpolated hair mask.
        
        Description:
            Uses the jaw and brows keypoints to figure out where the hair mask should be interpolated to on image 1
        """        
        
        # Loops through each y index and uses face keypoints to make sure hair isn't covering the face where its nto supposed to

        original_hair_mask = hair_mask2.clone()
        
        img1_left_points = mask1_npz["left_points"]
        img1_right_points = mask1_npz["right_points"]
        
        img2_left_points = mask2_npz["left_points"]
        img2_right_points = mask2_npz["right_points"]
        
        img_center = 256
        
        hair_mask2_left = original_hair_mask.clone()
        hair_mask2_left[:, :, img_center:] = 0
        
        hair_mask2_right = original_hair_mask.clone()
        hair_mask2_right[:, :, :img_center] = 0
        
        hair_left_points = img_center * 2 - torch.argmax(torch.flip(hair_mask2_left, dims=[2]), axis=2)[0]
        hair_right_points = torch.argmax(hair_mask2_right, axis=2)[0]

        img2_left_x_list, img2_right_x_list = self.create_x_points(img2_left_points, img2_right_points)
        
        img1_left_x_min, img1_left_x_max = img1_left_points.min(axis=0)[0], img1_left_points.max(axis=0)[0]
        img1_left_size = img1_left_x_max - img1_left_x_min + 1
        
        img1_right_x_min, img1_right_x_max = img1_right_points.min(axis=0)[0], img1_right_points.max(axis=0)[0]
        img1_right_size = img1_right_x_max - img1_right_x_min + 1

        # Fix relative size of the right and left size of the face
        for img1_i in range(img1_left_points.shape[0]):
            current_y = img1_left_points[img1_i, 1]
            # img2_i = int(img1_i / img1_left_points.shape[0] * img2_left_points.shape[0])

            if hair_left_points[current_y] != 0 and hair_left_points[current_y] != img_center * 2:
                x_perc = (img1_left_points[img1_i, 0] - img1_left_x_min) / img1_left_size
                img_2_x_idx = int(x_perc * img2_left_x_list.shape[0])
                img2_i = int(img2_left_x_list[img_2_x_idx] - img2_left_points[0, 1])
                
                left_hair_distance2 = hair_left_points[current_y] - img2_left_points[img2_i, 0]
                hair_move_point = left_hair_distance2 + img1_left_points[img1_i, 0]
                
                add_amount = hair_move_point - hair_left_points[current_y]
                if add_amount > 0:
                    hair_mask2[:, current_y, hair_left_points[current_y]: hair_left_points[current_y] + add_amount] = 1
                else:
                    hair_mask2[:, current_y, hair_left_points[current_y] + add_amount: hair_left_points[current_y]] = 0

            if hair_right_points[current_y] != 0 and hair_right_points[current_y] != img_center * 2:
                x_perc = (img1_right_points[img1_i, 0] - img1_right_x_min) / img1_right_size
                img_2_x_idx = int(x_perc * img2_right_x_list.shape[0])
                img2_i = int(img2_right_x_list[img_2_x_idx] - img2_right_points[0, 1])
                
                right_hair_distance2 = hair_right_points[current_y] - img2_right_points[img2_i, 0]
                hair_move_point = right_hair_distance2 + img1_right_points[img1_i, 0]

                add_amount = hair_right_points[current_y] - hair_move_point
                if add_amount > 0:
                    hair_mask2[:, current_y, hair_right_points[current_y] - add_amount: hair_right_points[current_y]] = 1
                else:
                    hair_mask2[:, current_y, hair_right_points[current_y]: hair_right_points[current_y] - add_amount] = 0
        
        # Loops through each x index to make sure bangs are in the right place
        img1_brows_points = mask1_npz["brows_points"]
        img2_brows_points = mask2_npz["brows_points"]
        
        hair_mask2_top = original_hair_mask.clone()
        hair_top_points = img_center * 2 - torch.argmax(torch.flip(hair_mask2_top, dims=[1]), axis=1)[0]
        for img1_i in range(img1_brows_points.shape[0]):
            current_x = img1_brows_points[img1_i, 0]

            img2_i = int(img1_i / img1_brows_points.shape[0] * img2_brows_points.shape[0])

            if hair_top_points[current_x] < img_center:
                top_hair_distance2 = hair_top_points[current_x] - img2_brows_points[img2_i, 1]
                hair_move_point = top_hair_distance2 + img1_brows_points[img1_i, 1]
                
                add_amount = hair_move_point - hair_top_points[current_x]
                if add_amount > 0:
                    hair_mask2[:, hair_top_points[current_x]: hair_move_point, current_x] = 1
                else:
                    hair_mask2[:, hair_move_point: hair_top_points[current_x], current_x] = 0
        
        return hair_mask2

    def create_x_points(self, img2_left_points: np.ndarray, img2_right_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
            img2_left_points (np.ndarray): An array of shape (n, 2) containing the x and y coordinates of the left points in `img2`.
            img2_right_points (np.ndarray): An array of shape (n, 2) containing the x and y coordinates of the right points in `img2`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays, `img2_left_x_list` and `img2_right_x_list`, that represent the x-coordinates of the left and right points in `img2_left_points` and `img2_right_points`, respectively.

        Description:
            This returns the y coordinates of each x index
            This means each index is a relative x index and elemment in the max y position with that x
            This is used to have a simple way to transalate the y coordinate of imag 1 to img2 based of the realative x coordinates
        """
         
        img2_left_x_min, img2_left_x_max = img2_left_points.min(axis=0)[0], img2_left_points.max(axis=0)[0]
        img2_right_x_min, img2_right_x_max = img2_right_points.min(axis=0)[0], img2_right_points.max(axis=0)[0]

        img2_left_x_list = np.ones((img2_left_x_max - img2_left_x_min + 1)) * -1
        img2_right_x_list = np.ones((img2_right_x_max - img2_right_x_min + 1)) * -1

        # instantiate left array with known
        for img2_i in range(img2_left_points.shape[0]):
            cur_x = img2_left_points[img2_i, 0]
            new_i = cur_x - img2_left_x_min
            img2_left_x_list[new_i] = img2_left_points[img2_i, 1]

        # instantiate right array with known
        for img2_i in range(img2_right_points.shape[0]):
            cur_x = img2_right_points[img2_i, 0]
            new_i = cur_x - img2_right_x_min
            img2_right_x_list[new_i] = img2_right_points[img2_i, 1]

        # Fill in the left holes if they are any
        previous_y = img2_left_x_list[0]
        for i in range(img2_left_x_list.shape[0]):
            if img2_left_x_list[i] == -1:
                img2_left_x_list[i] = previous_y
            previous_y = img2_left_x_list[i]

        # Fill in the right holes if they are any
        previous_y = img2_right_points[0]
        for i in range(img2_right_x_list.shape[0]):
            if img2_right_x_list[i] == -1:
                img2_right_x_list[i] = previous_y
            previous_y = img2_right_x_list[i]
            
        return img2_left_x_list, img2_right_x_list

    def setup_align_optimizer(self, cur_net: Model, latent_path: str=None, device="cuda"):
        """
        Set up the optimizer for the alignment process.

        Args:
            cur_net (object): The current network object.
            latent_path (str, optional): The path to the latent file. Defaults to None.
            device (str, optional): The device to use for the computation. Defaults to "cuda".

        Returns:
            tuple: A tuple containing the optimizer and the latent weight tensor.
        """
        
        if latent_path:
            latent_W = torch.from_numpy(convert_npy_code(np.load(latent_path))).to(device).requires_grad_(True)
        else:
            latent_W = cur_net.latent_avg.reshape(1, 1, 512).repeat(1, 18, 1).clone().detach().to(device).requires_grad_(True)
        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }

        optimizer_align = opt_dict[self.opts.opt_name]([latent_W], lr=self.opts.learning_rate)

        return optimizer_align, latent_W

    def create_down_seg(self, cur_seg: Model, cur_net: Model, latent_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a downsampled segmentation mask and synthesized image based on the given latent input.

        Args:
            cur_seg (nn.Module): The segmentation model used to generate the mask.
            cur_net (nn.Module): The generator network used to synthesize the image.
            latent_in (torch.Tensor): The latent input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the downsampled segmentation mask and the synthesized image.
        """
        
        gen_im, _ = cur_net.inference([latent_in], input_is_latent=True, return_latents=False,
                                       start_layer=0, end_layer=8)
        gen_im_0_1 = (gen_im + 1) / 2

        # get hair mask of synthesized image
        im = (self.downsample(gen_im_0_1) - cur_seg.user_seg_mean) / cur_seg.user_seg_std
        down_seg, _, _ = cur_seg(im)
        return down_seg, gen_im
        
    def dilate_erosion(self, free_mask: torch.Tensor, device, dilate_erosion: int=5) -> tuple[torch.Tensor, torch.Tensor]:
        free_mask = F.interpolate(free_mask.cpu(), size=(256, 256), mode='nearest').squeeze()
        free_mask_D, free_mask_E = cuda_unsqueeze(dilate_erosion_mask_tensor(free_mask, dilate_erosion=dilate_erosion), device)
        return free_mask_D, free_mask_E

    def get_hair_box(self, mask: torch.Tensor) -> tuple[int, int, int, int]:
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
    
    def align_images(self, img_path1: str, img_path2: str, sign='realistic', align_more_region: bool=False, smooth: int=5,
                     save_intermediate: bool=True) -> None:

        ################## img_path1: Identity Image
        ################## img_path2: Structure Image

        device = self.opts.device[0]
        output_dir = self.opts.output_dir
        target_mask, hair_mask_target, hair_mask1, hair_mask2 = \
            self.create_target_segmentation_mask(img_path1=img_path1, img_path2=img_path2, sign=sign,
                                                 save_intermediate=save_intermediate)
        
        im_name_1 = os.path.splitext(os.path.basename(img_path1))[0]
        im_name_2 = os.path.splitext(os.path.basename(img_path2))[0]

        latent_FS_path_1 = os.path.join(output_dir, 'FS', f'{im_name_1}.npz')
        latent_FS_path_2 = os.path.join(output_dir, 'FS', f'{im_name_2}.npz')

        latent_1, latent_F_1 = load_FS_latent(latent_FS_path_1, device)
        latent_2, latent_F_2 = load_FS_latent(latent_FS_path_2, device)

        latent_W_path_1 = os.path.join(output_dir, 'W+', f'{im_name_1}.npy')
        latent_W_path_2 = os.path.join(output_dir, 'W+', f'{im_name_2}.npy')

        def gpu_0_inference(
            results: dict, lock: Lock,
            cur_seg: Model, cur_net: Model, 
            quality_clip: Model, hair_classifier: Model,
            cur_loss_builder: AlignLossBuilder, cur_device
        ):
            torch.cuda.set_device(cur_net.device)
            
            optimizer_align, latent_align_1 = self.setup_align_optimizer(cur_net, latent_W_path_1, device=cur_device)

            cur_target_mask = target_mask.to(cur_device)

            quality_target_idx = None
            if quality_clip is not None:
                quality_target_idx = torch.tensor(0, dtype=cur_target_mask.dtype).to(cur_net.device)
            
            hair_class_idx = None
            if hair_classifier is not None:
                hair_class_idx = torch.tensor(self.opts.hair_class, dtype=cur_target_mask.dtype).to(cur_net.device)
            
            pbar = tqdm(range(self.opts.align_steps1), desc='Align Step 1', leave=False, disable=self.opts.disable_progress_bar)
            for step in pbar:
                optimizer_align.zero_grad()
                latent_in = torch.cat([latent_align_1[:, :6, :], latent_1[:, 6:, :]], dim=1)
                down_seg, gen_im = self.create_down_seg(cur_seg, cur_net, latent_in)
                
                loss_dict = {}
                
                # Cross Entropy Loss
                ce_loss = cur_loss_builder.cross_entropy_loss(down_seg, cur_target_mask)
                loss_dict["ce_loss"] = ce_loss.item()
                loss = ce_loss

                gen_im_0_1 = (gen_im + 1) / 2
                
                if quality_target_idx is not None and step >= self.opts.align_steps1 - self.opts.clip_quality_iterations:
                    model_in = quality_clip.torch_prepreocess(gen_im_0_1)
                    output = quality_clip.inference(model_in)[0]
                    quality_loss = cur_loss_builder.quality_cross_entropy_loss(output, quality_target_idx)
                    loss += quality_loss
                    loss_dict["quality_loss"] = quality_loss.item()
                    
                if hair_class_idx is not None and step >= self.opts.align_steps1 - self.opts.hair_classifier_iterations:
                    seg = down_seg.argmax(dim=1)
                    
                    output = hair_classifier.inference(gen_im_0_1, seg)[0]
                    hair_loss = cur_loss_builder.hair_cross_entropy_loss(output, hair_class_idx)
                    loss += hair_loss
                    loss_dict["hair_loss"] = hair_loss.item()

                loss.backward()
                optimizer_align.step()
    
            intermediate_align, _ = cur_net.inference([latent_in], input_is_latent=True, return_latents=False,
                                                       start_layer=0, end_layer=3)
            intermediate_align = intermediate_align.clone().detach()

            with lock:
                results["intermediate_align"] = intermediate_align
            
        ##############################################
        def gpu_1_inference(
            results: dict, lock: Lock, cur_seg: Model,
            cur_net: Model, cur_loss_builder: AlignLossBuilder, cur_device):
            torch.cuda.set_device(cur_net.device)
            
            cur_latent_2 = latent_2.to(cur_device)
            
            optimizer_align, latent_align_2 = self.setup_align_optimizer(cur_net, latent_W_path_2, device=cur_device)

            cur_target_mask = target_mask.to(cur_device)
            
            with torch.no_grad():
                tmp_latent_in = torch.cat([latent_align_2[:, :6, :], cur_latent_2[:, 6:, :]], dim=1)
                down_seg_tmp, I_Structure_Style_changed = self.create_down_seg(cur_seg, cur_net, tmp_latent_in)
    
                current_mask_tmp = torch.argmax(down_seg_tmp, dim=1).long()
                HM_Structure = torch.where(current_mask_tmp == 10, torch.ones_like(current_mask_tmp),
                                           torch.zeros_like(current_mask_tmp))
                HM_Structure = F.interpolate(HM_Structure.float().unsqueeze(0), size=(256, 256), mode='nearest')
    
            pbar = tqdm(range(self.opts.align_steps2), desc='Align Step 2', leave=False, disable=self.opts.disable_progress_bar)
            for step in pbar:
                optimizer_align.zero_grad()
                latent_in = torch.cat([latent_align_2[:, :6, :], cur_latent_2[:, 6:, :]], dim=1)
                down_seg, gen_im = self.create_down_seg(cur_seg, cur_net, latent_in)
    
                Current_Mask = torch.argmax(down_seg, dim=1).long()
                HM_G_512 = torch.where(Current_Mask == 10, torch.ones_like(Current_Mask),
                                       torch.zeros_like(Current_Mask)).float().unsqueeze(0)
                HM_G = F.interpolate(HM_G_512, size=(256, 256), mode='nearest')
    
                loss_dict = {}
                loss = 0
                # Segmentation Loss
                
                ce_loss = cur_loss_builder.cross_entropy_loss(down_seg, cur_target_mask)
                loss_dict["ce_loss"] = ce_loss.item()
                loss = ce_loss  # + hair_perc_loss
    
                # Style Loss
                H1_region = self.downsample_256(I_Structure_Style_changed) * HM_Structure
                H2_region = self.downsample_256(gen_im) * HM_G
                style_loss = cur_loss_builder.style_loss(H1_region, H2_region, mask1=HM_Structure, mask2=HM_G)
    
                loss_dict["style_loss"] = style_loss.item()
                loss += style_loss
                
                loss.backward()
                optimizer_align.step()

            latent_F_out_new, _ = cur_net.inference([latent_in], input_is_latent=True, return_latents=False,
                                                     start_layer=0, end_layer=3)
            latent_F_out_new = latent_F_out_new.clone().detach()

            with lock:
                results["latent_F_out_new"] = latent_F_out_new

        # Run Threads
        results = {}
        threading_lock = Lock()

        if self.opts.is_multi_gpu:
            print("Multi-threading aligment")
            gpu0_thread = Thread(
                target=gpu_0_inference,
                args=(results, threading_lock, self.seg0, self.net0, self.quality_clip, self.hair_classifier, self.loss_builder0, self.opts.device[0])
            )
            gpu1_thread = Thread(
                target=gpu_1_inference,
                args=(results, threading_lock, self.seg1, self.net1, self.loss_builder1, self.opts.device[1])
            )

            gpu1_thread.start()
            gpu0_thread.start()
            gpu0_thread.join()
            gpu1_thread.join()
        else:
            print("Single threading aligment")
            gpu_0_inference(results, threading_lock, self.seg0, self.net0, self.quality_clip, self.loss_builder0, cur_device=self.opts.device[0])
            gpu_1_inference(results, threading_lock, self.seg0, self.net0, self.loss_builder0, cur_device=self.opts.device[0])
            
        # Loads results
        intermediate_align = results["intermediate_align"]
        latent_F_out_new = results["latent_F_out_new"].to(self.opts.device[0])
        
        free_mask = 1 - (1 - hair_mask1.unsqueeze(0)) * (1 - hair_mask_target)

        ##############################
        free_mask, _ = self.dilate_erosion(free_mask, device, dilate_erosion=smooth)
        ##############################

        free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode='bicubic')[0]
        interpolation_low = 1 - free_mask_down_32
        
        latent_F_mixed = intermediate_align + interpolation_low.unsqueeze(0) * (
                latent_F_1 - intermediate_align)

        if not align_more_region:
            free_mask = hair_mask_target
            ##########################
            _, free_mask = self.dilate_erosion(free_mask, device, dilate_erosion=smooth)
            ##########################
            free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode='bicubic')[0]
            interpolation_low = 1 - free_mask_down_32

        latent_F_mixed = latent_F_out_new + interpolation_low.unsqueeze(0) * (
                latent_F_mixed - latent_F_out_new)

        free_mask = F.interpolate((hair_mask2.unsqueeze(0) * hair_mask_target).float(), size=(256, 256), mode='nearest').cuda()
        ##########################
        _, free_mask = self.dilate_erosion(free_mask, device, dilate_erosion=smooth)
        ##########################
        free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode='bicubic')[0]
        interpolation_low = 1 - free_mask_down_32

        latent_F_mixed = latent_F_2 + interpolation_low.unsqueeze(0) * (
                latent_F_mixed - latent_F_2)

        gen_im, _ = self.net0.generator([latent_1], input_is_latent=True, return_latents=False, start_layer=4,
                                       end_layer=8, layer_in=latent_F_mixed)
        
        self.save_align_results(im_name_1, im_name_2, sign, gen_im, latent_1, latent_F_mixed,
                                save_intermediate=save_intermediate)

    def save_align_results(self, im_name_1: str, im_name_2: str, sign: str, gen_im: torch.Tensor, latent_in: torch.Tensor, latent_F: torch.Tensor, save_intermediate: bool=True):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))

        save_dir = os.path.join(self.opts.output_dir, 'Align_{}'.format(sign))
        os.makedirs(save_dir, exist_ok=True)

        latent_path = os.path.join(save_dir, '{}_{}.npz'.format(im_name_1, im_name_2))
        if save_intermediate:
            image_path = os.path.join(save_dir, '{}_{}.png'.format(im_name_1, im_name_2))
            save_im.save(image_path)
        np.savez(latent_path, latent_in=latent_in.detach().cpu().numpy(), latent_F=latent_F.detach().cpu().numpy())
