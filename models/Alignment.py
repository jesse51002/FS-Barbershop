import torch
from torch import nn
import numpy as np
import os
from functools import partial
from utils.bicubic import BicubicDownSample
from tqdm import tqdm
import torchvision
from utils.data_utils import convert_npy_code
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from models.face_parsing.classes import CLASSES
from losses.align_loss import AlignLossBuilder
import torch.nn.functional as F
import cv2
from utils.data_utils import load_FS_latent
from utils.seg_utils import save_vis_mask
from utils.model_utils import download_weight
from utils.data_utils import cuda_unsqueeze
from utils.image_utils import dilate_erosion_mask_tensor
from threading import Thread, Lock

toPIL = torchvision.transforms.ToPILImage()


class Alignment(nn.Module):

    def __init__(self, opts, net0=None, net1=None, seg=None):
        super(Alignment, self).__init__()
        
        self.opts = opts
        self.net0 = net0
        self.net1 = net1
        self.seg = seg

        self.load_downsampling()
        self.setup_align_loss_builder()

    def set_opts(self, opts):
        self.opts = opts

    def load_downsampling(self):
        self.downsample = BicubicDownSample(factor=self.opts.size // 512)
        self.downsample_256 = BicubicDownSample(factor=self.opts.size // 256)

    def setup_align_loss_builder(self):
        self.loss_builder0 = AlignLossBuilder(self.opts, num_classes=len(CLASSES), device=self.opts.device[0])
        if self.opts.is_multi_gpu:
            self.loss_builder1 = AlignLossBuilder(self.opts, num_classes=len(CLASSES), device=self.opts.device[1])
            
    def create_target_segmentation_mask(self, img_path1, img_path2, sign, save_intermediate=True):
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
        
        # --------------------------------------------
        # Loops through each y index and uses face keypoints to make sure hair isn't covering the face where its nto supposed to
        img1_left_points = mask1_npz["left_points"]
        img1_right_points = mask1_npz["right_points"]

        img2_left_points = mask2_npz["left_points"]
        img2_right_points = mask2_npz["right_points"]

        img_center = 256
        
        hair_mask2_left = hair_mask2.clone()
        hair_mask2_left[:, :, img_center:] = 0
        
        hair_mask2_right = hair_mask2.clone()
        hair_mask2_right[:, :, :img_center] = 0
        
        hair_left_points = img_center * 2 - torch.argmax(torch.flip(hair_mask2_left, dims=[2]), axis=2)[0]
        hair_right_points = torch.argmax(hair_mask2_right, axis=2)[0]
        
        for img1_i in range(img1_left_points.shape[0]):
            current_y = img1_left_points[img1_i, 1]

            img2_i = int(img1_i / img1_left_points.shape[0] * img2_left_points.shape[0])

            if img1_i < 0 or img1_i >= img1_left_points.shape[0]:
                continue

            if hair_left_points[current_y] != 0:
                left_hair_distance2 = hair_left_points[current_y] - img2_left_points[img2_i, 0]
                hair_move_point = left_hair_distance2 + img1_left_points[img1_i, 0]
                
                add_amount = hair_move_point - hair_left_points[current_y]
                if add_amount > 0:
                    hair_mask2[:, current_y, hair_left_points[current_y]: hair_left_points[current_y] + add_amount] = 1
                else:
                    hair_mask2[:, current_y, hair_left_points[current_y] + add_amount: hair_left_points[current_y]] = 0

            if hair_right_points[current_y] != 0:
                right_hair_distance2 = hair_right_points[current_y] - img2_right_points[img2_i, 0]
                hair_move_point = right_hair_distance2 + img1_right_points[img1_i, 0]

                add_amount = hair_right_points[current_y] - hair_move_point
                if add_amount > 0:
                    hair_mask2[:, current_y, hair_right_points[current_y] - add_amount: hair_right_points[current_y]] = 1
                else:
                    hair_mask2[:, current_y, hair_right_points[current_y]: hair_right_points[current_y] - add_amount] = 0

        # Adds the hair to the mask
        seg_target2 = torch.where((hair_mask2 == 0) & (seg_target2 == CLASSES["hair"]), 0, seg_target2)
        seg_target2 = torch.where(hair_mask2 == 1, CLASSES["hair"], seg_target2)
        hair_mask2 = torch.where(seg_target2 == CLASSES["hair"], torch.ones_like(seg_target2), torch.zeros_like(seg_target2))
        
        # ----------------------------------------------

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
        right, left, bottom, top = self.get_hair_box(target_mask)
        
        # Crops the box inwards to not mess up the curve on the outer hair curve
        top_crop_perc = 0.3
        side_crop_perc = 0.4

        top_move_amount = int(top_crop_perc * (top - bottom))
        side_move_amount = int(side_crop_perc / 2 * (left - right))

        right += side_move_amount
        left -= side_move_amount
        bottom += top_move_amount

        binary_box_mask = torch.zeros_like(target_mask)
        binary_box_mask[:, bottom:top, right:left] = 1

        target_mask = torch.where((binary_box_mask == 1) & (target_mask == 0), CLASSES["hair"], target_mask)
         
        """
        # Adding temporaily inorder to fix mask issues --------------------------------------------------------------------------

        ############################# add auto-inpainting

        optimizer_align, latent_align = self.setup_align_optimizer()
        latent_end = latent_align[:, 6:, :].clone().detach()

        pbar = tqdm(range(80), desc='Create Target Mask Step1', leave=False)
        for step in pbar:
            optimizer_align.zero_grad()
            latent_in = torch.cat([latent_align[:, :6, :], latent_end], dim=1)
            down_seg, _ = self.create_down_seg(latent_in)
            loss_dict = {}
            
            if sign == 'realistic':
                ce_loss = self.loss_builder.cross_entropy_loss_wo_background(down_seg, target_mask)
                ce_loss += self.loss_builder.cross_entropy_loss_only_background(down_seg, ggg)
            else:
                ce_loss = self.loss_builder.cross_entropy_loss(down_seg, target_mask)

            loss_dict["ce_loss"] = ce_loss.item()
            loss = ce_loss

            loss.backward()
            optimizer_align.step()

        gen_seg_target = torch.argmax(down_seg, dim=1).long()
        free_mask = hair_mask1 * (1 - hair_mask2)
        target_mask = torch.where(free_mask == 1, gen_seg_target, target_mask)
        previouse_target_mask = target_mask.clone().detach()

        ############################################

        target_mask = torch.where(OB_region.to(device).unsqueeze(0), torch.zeros_like(target_mask), target_mask)
        optimizer_align, latent_align = self.setup_align_optimizer()
        latent_end = latent_align[:, 6:, :].clone().detach()

        pbar = tqdm(range(80), desc='Create Target Mask Step2', leave=False)
        for step in pbar:
            optimizer_align.zero_grad()
            latent_in = torch.cat([latent_align[:, :6, :], latent_end], dim=1)
            down_seg, _ = self.create_down_seg(latent_in)

            loss_dict = {}

            if sign == 'realistic':
                ce_loss = self.loss_builder.cross_entropy_loss_wo_background(down_seg, target_mask)
                ce_loss += self.loss_builder.cross_entropy_loss_only_background(down_seg, ggg)
            else:
                ce_loss = self.loss_builder.cross_entropy_loss(down_seg, target_mask)

            loss_dict["ce_loss"] = ce_loss.item()
            loss = ce_loss

            loss.backward()
            optimizer_align.step()

        gen_seg_target = torch.argmax(down_seg, dim=1).long()
        # free_mask = hair_mask1 * (1 - hair_mask2)
        # target_mask = torch.where((free_mask == 1) * (gen_seg_target!=0), gen_seg_target, previouse_target_mask)
        target_mask = torch.where((OB_region.to(device).unsqueeze(0)) * (gen_seg_target != 0), gen_seg_target, previouse_target_mask)
        
        # ------------------------------------------------------
        """

        
        """
        # This is for a loss that also takes into account the percentage of the area being hair
        # This will help with hair that fades in and out of the background (prolly)
        # it keeps the hair percentages and keeps everything else as binary
        target_hair_percentages = torch.zeros((1, *target_mask.shape[-2:])).long().cuda()
        target_hair_percentages[0] = down_seg2[0, 10]
        """
        
        # Visualize the hair
        # plt.imshow(target_mask_hair_percentages[0, 10].clone().cpu().numpy())
        # plt.savefig("./test.png")

        hair_mask_target = torch.where(target_mask == CLASSES["hair"], torch.ones_like(target_mask), torch.zeros_like(target_mask))
        hair_mask_target = hair_mask_target.float().unsqueeze(0)
        
        # ---------------  Save Visualization of Target Segmentation Mask
        masks_dir = os.path.join(self.opts.output_dir, "masks")
        if save_intermediate:
            save_vis_mask(img_path1, img_path2, sign, masks_dir, target_mask.squeeze().cpu())

        return target_mask, hair_mask_target, hair_mask1, hair_mask2
        
    def setup_align_optimizer(self, cur_net, latent_path=None, device="cuda"):
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

    def create_down_seg(self, cur_net, latent_in, face_mode=True):
        gen_im, _ = cur_net.generator([latent_in], input_is_latent=True, return_latents=False,
                                       start_layer=0, end_layer=8)
        gen_im_0_1 = (gen_im + 1) / 2

        # get hair mask of synthesized image
        im = (self.downsample(gen_im_0_1) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(im)
        return down_seg, gen_im
        
    def dilate_erosion(self, free_mask, device, dilate_erosion=5):
        free_mask = F.interpolate(free_mask.cpu(), size=(256, 256), mode='nearest').squeeze()
        free_mask_D, free_mask_E = cuda_unsqueeze(dilate_erosion_mask_tensor(free_mask, dilate_erosion=dilate_erosion), device)
        return free_mask_D, free_mask_E

    def get_hair_box(self, mask):
        right, left, bottom, top = 0, 0, 0, 0
        # Gets positions where the data is not 0
        contains_pos = torch.argwhere(mask == CLASSES["hair"])
        
        # Handles empty case
        if contains_pos.shape[0] == 0:
            return right, left, bottom, top
        
        # min index where element is not zero
        mins = torch.min(contains_pos, axis=0).values
        bottom = mins[1]
        right = mins[2]
        
        # Max index where element is not zero
        maxs = torch.max(contains_pos, axis=0).values
        top = maxs[1]
        left = maxs[2]

        return right, left, bottom, top
    
    def align_images(self, img_path1, img_path2, sign='realistic', align_more_region=False, smooth=5,
                     save_intermediate=True):

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

        def gpu_0_inference(results, lock, cur_net, cur_loss_builder, cur_device):
            optimizer_align, latent_align_1 = self.setup_align_optimizer(cur_net, latent_W_path_1)

            cur_target_mask = target_mask.to(cur_device)
            
            pbar = tqdm(range(self.opts.align_steps1), desc='Align Step 1', leave=False)
            for step in pbar:
                face_loss = step % self.opts.body_alternate_number != 0
                
                optimizer_align.zero_grad()
                latent_in = torch.cat([latent_align_1[:, :6, :], latent_1[:, 6:, :]], dim=1)
                down_seg, _ = self.create_down_seg(cur_net, latent_in, face_mode=face_loss)
    
                loss_dict = {}
                loss = 0
                
                # Cross Entropy Loss
                ce_loss = cur_loss_builder.cross_entropy_loss(down_seg, cur_target_mask)
                loss_dict["ce_loss"] = ce_loss.item()
                loss = ce_loss
    
                loss.backward()
                optimizer_align.step()
    
            intermediate_align, _ = cur_net.generator([latent_in], input_is_latent=True, return_latents=False,
                                                       start_layer=0, end_layer=3)
            intermediate_align = intermediate_align.clone().detach()

            with lock:
                results["intermediate_align"] = intermediate_align
            
        ##############################################
        def gpu_1_inference(results, lock, cur_net, cur_loss_builder, cur_device):
            cur_latent_2 = latent_2.to(cur_device)
            
            optimizer_align, latent_align_2 = self.setup_align_optimizer(cur_net, latent_W_path_2)

            cur_target_mask = target_mask.to(cur_device)
            
            with torch.no_grad():
                tmp_latent_in = torch.cat([latent_align_2[:, :6, :], cur_latent_2[:, 6:, :]], dim=1)
                down_seg_tmp, I_Structure_Style_changed = self.create_down_seg(cur_net, tmp_latent_in)
    
                current_mask_tmp = torch.argmax(down_seg_tmp, dim=1).long()
                HM_Structure = torch.where(current_mask_tmp == 10, torch.ones_like(current_mask_tmp),
                                           torch.zeros_like(current_mask_tmp))
                HM_Structure = F.interpolate(HM_Structure.float().unsqueeze(0), size=(256, 256), mode='nearest')
    
            pbar = tqdm(range(self.opts.align_steps2), desc='Align Step 2', leave=False)
            for step in pbar:
                face_loss = step % self.opts.body_alternate_number != 0
                
                optimizer_align.zero_grad()
                latent_in = torch.cat([latent_align_2[:, :6, :], cur_latent_2[:, 6:, :]], dim=1)
                down_seg, gen_im = self.create_down_seg(cur_net, latent_in, face_mode=face_loss)
    
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

            latent_F_out_new, _ = cur_net.generator([latent_in], input_is_latent=True, return_latents=False,
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
                args=(results, threading_lock, self.net0, self.loss_builder0, self.opts.device[0])
            )
            gpu1_thread = Thread(
                target=gpu_1_inference,
                args=(results, threading_lock, self.net1, self.loss_builder1, self.opts.device[1])
            )
    
            gpu0_thread.start()
            gpu1_thread.start()
            gpu0_thread.join()
            gpu1_thread.join()
        else:
            print("Single threading aligment")
            gpu_0_inference(results, threading_lock, self.net0, self.loss_builder0, cur_device=self.opts.device[0])
            gpu_1_inference(results, threading_lock, self.net0, self.loss_builder0, cur_device=self.opts.device[0])
            
        # Loads results
        intermediate_align = results["intermediate_align"]
        latent_F_out_new = results["latent_F_out_new"]
        
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

    def save_align_results(self, im_name_1, im_name_2, sign, gen_im, latent_in, latent_F, save_intermediate=True):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))

        save_dir = os.path.join(self.opts.output_dir, 'Align_{}'.format(sign))
        os.makedirs(save_dir, exist_ok=True)

        latent_path = os.path.join(save_dir, '{}_{}.npz'.format(im_name_1, im_name_2))
        if save_intermediate:
            image_path = os.path.join(save_dir, '{}_{}.png'.format(im_name_1, im_name_2))
            save_im.save(image_path)
        np.savez(latent_path, latent_in=latent_in.detach().cpu().numpy(), latent_F=latent_F.detach().cpu().numpy())
