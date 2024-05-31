import torch
from torch import nn
import numpy as np
import os
from utils.bicubic import BicubicDownSample
from tqdm import tqdm
import torchvision
from models.face_parsing.model import BiSeNet
from models.face_parsing.classes import CLASSES
from models.optimizer.ClampOptimizer import ClampOptimizer
from losses.blend_loss import BlendLossBuilder
import torch.nn.functional as F
from utils.data_utils import load_FS_latent
from utils.data_utils import cuda_unsqueeze
from utils.image_utils import load_image, dilate_erosion_mask_path, dilate_erosion_mask_tensor
from utils.model_utils import download_weight
from utils.seg_utils import expand_face_mask
from models.FacerParsing import facer_to_bisnet

toPIL = torchvision.transforms.ToPILImage()


class Blending(nn.Module):
    def __init__(self, opts, net=None, facer=None, background_remover=None, seg=None):
        super(Blending, self).__init__()
        
        self.opts = opts
        self.net = net
        self.facer = facer
        self.background_remover = background_remover
        self.seg = seg
        
        self.load_downsampling()
        self.setup_blend_loss_builder()

    def set_opts(self, opts):
        self.opts = opts
    
    def load_downsampling(self):
        self.downsample = BicubicDownSample(factor=self.opts.size // 512)
        self.downsample_256 = BicubicDownSample(factor=self.opts.size // 256)

    def setup_blend_optimizer(self):
        interpolation_latent = torch.zeros((18, 512), requires_grad=True, device=self.opts.device[0])

        opt_blend = ClampOptimizer(torch.optim.Adam, [interpolation_latent], lr=self.opts.learning_rate)

        return opt_blend, interpolation_latent

    def setup_blend_loss_builder(self):
        self.loss_builder = BlendLossBuilder(self.opts.device[0])

    def blend_images(self, img_path1, img_path2, img_path3, sign='realistic'):
        device = self.opts.device[0]
        output_dir = self.opts.output_dir

        im_name_1 = os.path.splitext(os.path.basename(img_path1))[0]
        im_name_2 = os.path.splitext(os.path.basename(img_path2))[0]
        im_name_3 = os.path.splitext(os.path.basename(img_path3))[0]

        I_1 = load_image(img_path1, downsample=True).to(device).unsqueeze(0)
        I_3 = load_image(img_path3, downsample=True).to(device).unsqueeze(0)

        HM_1D, _ = cuda_unsqueeze(dilate_erosion_mask_path(self.opts, img_path1), device)
        HM_3D, HM_3E = cuda_unsqueeze(dilate_erosion_mask_path(self.opts, img_path3), device)

        opt_blend, interpolation_latent = self.setup_blend_optimizer()
        latent_1, latent_F_mixed = load_FS_latent(
            os.path.join(output_dir, 'Align_{}'.format(sign), '{}_{}.npz'.format(im_name_1, im_name_3)),
            device
        )
        latent_3, _ = load_FS_latent(
            os.path.join(output_dir, 'FS', '{}.npz'.format(im_name_3)), device
        )

        with torch.no_grad():
            I_X, _ = self.net.generator(
                [latent_1], input_is_latent=True, return_latents=False, start_layer=4,
                end_layer=8, layer_in=latent_F_mixed
            )
            I_X_0_1 = (I_X + 1) / 2
            IM = I_X_0_1 * 255
            
            seg_targets = facer_to_bisnet(self.facer.inference(IM)[0]).float().detach().cpu().numpy()
            human_segs = self.background_remover.inference(IM)[1].detach().cpu().numpy()
            
            current_mask = torch.tensor(expand_face_mask(seg_targets[0], human_segs[0])).unsqueeze(0).float().to()
            
            HM_X = torch.where(current_mask == CLASSES["hair"], torch.ones_like(current_mask), torch.zeros_like(current_mask))
            HM_X = F.interpolate(HM_X.unsqueeze(0), size=(256, 256), mode='nearest').squeeze()
            HM_XD, _ = cuda_unsqueeze(dilate_erosion_mask_tensor(HM_X), device)
            target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        pbar = tqdm(range(self.opts.blend_steps), desc='Blend', leave=False, disable=self.opts.disable_progress_bar)
        for step in pbar:

            opt_blend.zero_grad()

            latent_mixed = latent_1 + interpolation_latent.unsqueeze(0) * (latent_3 - latent_1)

            I_G, _ = self.net.generator([latent_mixed], input_is_latent=True, return_latents=False, start_layer=4,
                               end_layer=8, layer_in=latent_F_mixed)
            I_G_0_1 = (I_G + 1) / 2

            im_dict = {
                'gen_im': self.downsample_256(I_G),
                'im_1': I_1,
                'im_3': I_3,
                'mask_face': target_mask,
                'mask_hair': HM_3E
            }
            
            loss, loss_dic = self.loss_builder(**im_dict)

            # if self.opts.verbose:
            #     pbar.set_description(
            #         'Blend Loss: {:.3f}, face: {:.3f}, hair: {:.3f}'
            #             .format(loss, loss_dic['face'], loss_dic['hair']))

            loss.backward()
            opt_blend.step()

        ############## Load F code from  '{}_{}.npz'.format(im_name_1, im_name_2)
        _, latent_F_mixed = load_FS_latent(os.path.join(output_dir, 'Align_{}'.format(sign),
                                                        '{}_{}.npz'.format(im_name_1, im_name_2)), device)
        I_G, _ = self.net.generator([latent_mixed], input_is_latent=True, return_latents=False, start_layer=4,
                           end_layer=8, layer_in=latent_F_mixed)

        self.save_blend_results(im_name_1, im_name_2, im_name_3, sign, I_G, latent_mixed, latent_F_mixed)

    def save_blend_results(self, im_name_1, im_name_2, im_name_3, sign,  gen_im, latent_in, latent_F):
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))

        save_dir = os.path.join(self.opts.output_dir, 'Blend_{}'.format(sign))
        os.makedirs(save_dir, exist_ok=True)

        latent_path = os.path.join(save_dir, '{}_{}_{}.npz'.format(im_name_1, im_name_2, im_name_3))
        image_path = os.path.join(save_dir, '{}_{}_{}.png'.format(im_name_1, im_name_2, im_name_3))
        output_image_path = os.path.join(self.opts.output_dir, '{}_{}_{}_{}.png'.format(im_name_1, im_name_2, im_name_3, sign))

        save_im.save(image_path)
        save_im.save(output_image_path)
        np.savez(latent_path, latent_in=latent_in.detach().cpu().numpy(), latent_F=latent_F.detach().cpu().numpy())


