import torch
from torch import nn
import numpy as np
import os
from functools import partial
from utils.bicubic import BicubicDownSample
from datasets.image_dataset import ImagesDataset
from losses.embedding_loss import EmbeddingLossBuilder
from torch.utils.data import DataLoader
from tqdm import tqdm
import PIL
import torchvision
from utils.data_utils import convert_npy_code
from models.ModelBase import Model

toPIL = torchvision.transforms.ToPILImage()


class Embedding(nn.Module):
    def __init__(self, opts: dict, net: Model):
        """
        Initializes the Embedding class.

        Args:
            opts (object): An object containing the options for the embedding.
            net (object): An object representing the neural network.

        Returns:
            None
        """
        
        super(Embedding, self).__init__()
        
        self.opts = opts
        self.net = net
        
        self.load_downsampling()
        self.setup_embedding_loss_builder()

    def set_opts(self, opts: dict) -> None:
        self.opts = opts

    def load_downsampling(self) -> None:
        factor = self.opts.size // 256
        self.downsample = BicubicDownSample(factor=factor)
        
    def setup_W_optimizer(self):
        """
        Initializes the optimizer for the W parameter in the embedding model.

        Returns:
            optimizer_W (torch.optim.Optimizer): The optimizer for the W parameter.
            latent (List[torch.Tensor]): The list of latent tensors.
        """
        
        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        latent = []
        if (self.opts.tile_latent):
            tmp = self.net.latent_avg.clone().detach().to(self.opts.device[0])
            tmp.requires_grad = True
            for i in range(self.net.layer_num):
                latent.append(tmp)
            optimizer_W = opt_dict[self.opts.opt_name]([tmp], lr=self.opts.learning_rate)
        else:
            for i in range(self.net.layer_num):
                tmp = self.net.latent_avg.clone().detach().to(self.opts.device[0])
                tmp.requires_grad = True
                latent.append(tmp)
            optimizer_W = opt_dict[self.opts.opt_name](latent, lr=self.opts.learning_rate)

        return optimizer_W, latent

    def setup_FS_optimizer(self, latent_W: torch.Tensor, F_init: torch.Tensor):
        """
        Sets up the optimizer for the FS model.

        Args:
            latent_W (torch.Tensor): The latent tensor for the W parameter.
            F_init (torch.Tensor): The initial tensor for the F parameter.

        Returns:
            optimizer_FS (torch.optim.Optimizer): The optimizer for the FS model.
            latent_F (torch.Tensor): The latent tensor for the F parameter.
            latent_S (List[torch.Tensor]): The list of latent tensors for the S parameter.
        """

        latent_F = F_init.clone().detach().requires_grad_(True)
        latent_S = []
        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        for i in range(self.net.layer_num):

            tmp = latent_W[0, i].clone()

            if i < self.net.S_index:
                tmp.requires_grad = False
            else:
                tmp.requires_grad = True

            latent_S.append(tmp)

        optimizer_FS = opt_dict[self.opts.opt_name](latent_S[self.net.S_index:] + [latent_F], lr=self.opts.learning_rate)

        return optimizer_FS, latent_F, latent_S

    def setup_dataloader(self, image_path: str=None) -> None:
        """
        Sets up the dataloader for the current model.

        Parameters:
            image_path (str, optional): The path to the image dataset. Defaults to None.

        Returns:
            None
        """

        self.dataset = ImagesDataset(opts=self.opts, image_path=image_path)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        print("Number of images: {}".format(len(self.dataset)))

    def setup_embedding_loss_builder(self) -> None:
        self.loss_builder = EmbeddingLossBuilder(self.opts, device=self.net.device)

    def invert_images_in_W(self, image_path: list[str]) -> None:
        """
        Inverts images in the W parameter of the embedding model.

        Args:
            image_path (list[str]): A list of paths to the images to be inverted.

        Returns:
            None
            
        Description:
            This function inverts images in the W parameter of the embedding model.
            Saves the inverted images to the 'W+' folder in the output directory.
        """
        
        image_path = image_path.copy()
        # Causes images that already have saved data to be ignored
        cur_idx = 0
        while cur_idx < len(image_path):
            base = os.path.basename(image_path[cur_idx]).split(".")[0]
            
            output_dir = os.path.join(self.opts.output_dir, 'W+')
            latent_path = os.path.join(output_dir, f'{base}.npy')
            dis_image_path = os.path.join(output_dir, f'{base}.png')
            
            if os.path.isfile(latent_path) and os.path.isfile(dis_image_path):
                image_path.pop(cur_idx)
            else:
                cur_idx += 1
                
        if len(image_path) == 0:
            return
            
        self.setup_dataloader(image_path=image_path)
        device = self.opts.device[0]
        ibar = tqdm(self.dataloader, desc='Images', disable=self.opts.disable_progress_bar)
        for ref_im_H, ref_im_L, ref_name in ibar:
            optimizer_W, latent = self.setup_W_optimizer()
            pbar = tqdm(range(self.opts.W_steps), desc='Embedding', leave=False, disable=self.opts.disable_progress_bar)
            for step in pbar:
                optimizer_W.zero_grad()
                latent_in = torch.stack(latent).unsqueeze(0)

                gen_im, _ = self.net.inference([latent_in], input_is_latent=True, return_latents=False)
                im_dict = {
                    'ref_im_H': ref_im_H.to(device),
                    'ref_im_L': ref_im_L.to(device),
                    'gen_im_H': gen_im,
                    'gen_im_L': self.downsample(gen_im)
                }

                loss, loss_dic = self.cal_loss(im_dict, latent_in)
                loss.backward()
                optimizer_W.step()

                if self.opts.verbose:
                    pbar.set_description('Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}'
                                         .format(loss, loss_dic['l2'], loss_dic['percep'], loss_dic['p-norm']))

                if self.opts.save_intermediate and step % self.opts.save_interval== 0:
                    self.save_W_intermediate_results(ref_name, gen_im, latent_in, step)

            self.save_W_results(ref_name, gen_im, latent_in)

    def invert_images_in_FS(self, image_path: list[str]) -> None:
        """
        Inverts images in the W parameter of the embedding model.

        Args:
            image_path (list[str]): A list of paths to the images to be inverted.

        Returns:
            None
            
        Description:
            This function inverts images in the W parameter of the embedding model.
            Saves the inverted images to the 'FS' folder in the output directory.
        """
        
        image_path = image_path.copy()
        # Causes images that already have saved data to be ignored
        cur_idx = 0
        while cur_idx < len(image_path):
            base = os.path.basename(image_path[cur_idx]).split(".")[0]
            
            output_dir = os.path.join(self.opts.output_dir, 'FS')
            latent_path = os.path.join(output_dir, f'{base}.npz')
            dis_image_path = os.path.join(output_dir, f'{base}.png')
            
            if os.path.isfile(latent_path) and os.path.isfile(dis_image_path):
                image_path.pop(cur_idx)
            else:
                cur_idx += 1

        if len(image_path) == 0:
            return
        
        self.setup_dataloader(image_path=image_path)
        output_dir = self.opts.output_dir
        device = self.opts.device[0]
        ibar = tqdm(self.dataloader, desc='Images', disable=self.opts.disable_progress_bar)
        for ref_im_H, ref_im_L, ref_name in ibar:

            latent_W_path = os.path.join(output_dir, 'W+', f'{ref_name[0]}.npy')
            latent_W = torch.from_numpy(convert_npy_code(np.load(latent_W_path))).to(device)
            F_init, _ = self.net.inference([latent_W], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)
            optimizer_FS, latent_F, latent_S = self.setup_FS_optimizer(latent_W, F_init)

            pbar = tqdm(range(self.opts.FS_steps), desc='Embedding', leave=False, disable=self.opts.disable_progress_bar)
            for step in pbar:

                optimizer_FS.zero_grad()
                latent_in = torch.stack(latent_S).unsqueeze(0)
                gen_im, _ = self.net.inference([latent_in], input_is_latent=True, return_latents=False,
                                               start_layer=4, end_layer=8, layer_in=latent_F)
                im_dict = {
                    'ref_im_H': ref_im_H.to(device),
                    'ref_im_L': ref_im_L.to(device),
                    'gen_im_H': gen_im,
                    'gen_im_L': self.downsample(gen_im)
                }

                loss, loss_dic = self.cal_loss(im_dict, latent_in)
                loss.backward()
                optimizer_FS.step()

                if self.opts.verbose:
                    pbar.set_description(
                        'Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}, L_F loss: {:.3f}'
                        .format(loss, loss_dic['l2'], loss_dic['percep'], loss_dic['p-norm'], loss_dic['l_F']))

            self.save_FS_results(ref_name, gen_im, latent_in, latent_F)

    def cal_loss(self, im_dict: dict, latent_in: torch.Tensor, latent_F: torch.Tensor, F_init: torch.Tensor):
        """
        Calculates the loss for the given image dictionary and latent inputs.

        Args:
            im_dict (dict): A dictionary containing the reference image, generated image, and other image information.
            latent_in (torch.Tensor): The input latent tensor.
            latent_F (torch.Tensor): The latent tensor for F.
            F_init (torch.Tensor): The initial latent tensor for F.

        Returns:
            tuple: A tuple containing the total loss and a dictionary of individual loss components.
        """
        
        loss, loss_dic = self.loss_builder(**im_dict)
        p_norm_loss = self.net.cal_p_norm_loss(latent_in)
        loss_dic['p-norm'] = p_norm_loss
        loss += p_norm_loss

        if latent_F is not None and F_init is not None:
            l_F = self.net.cal_l_F(latent_F, F_init)
            loss_dic['l_F'] = l_F
            loss += l_F

        return loss, loss_dic

    def save_W_results(self, ref_name, gen_im: torch.Tensor, latent_in: torch.Tensor)  -> None:
        """
        Save the generated image and latent tensor to the output directory.
        Args:
            ref_name (str): The reference name for the saved image and latent tensor.
            gen_im (torch.Tensor): The generated image tensor.
            latent_in (torch.Tensor): The input latent tensor.
        Returns:
            None
        
        Description:
            Saves the generated image and latent tensor to the output directory.
        """
        
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()

        output_dir = os.path.join(self.opts.output_dir, 'W+')
        os.makedirs(output_dir, exist_ok=True)

        latent_path = os.path.join(output_dir, f'{ref_name[0]}.npy')
        image_path = os.path.join(output_dir, f'{ref_name[0]}.png')

        save_im.save(image_path)
        np.save(latent_path, save_latent)

    def save_W_intermediate_results(self, ref_name, gen_im: torch.Tensor, latent_in: torch.Tensor, step: int) -> None:
        """
        Save the intermediate results of the W parameter of the embedding model.

        Args:
            ref_name (str): The reference name for the saved image and latent tensor.
            gen_im (torch.Tensor): The generated image tensor.
            latent_in (torch.Tensor): The input latent tensor.
            step (int): The step number of the intermediate results.

        Returns:
            None

        Description:
            This function saves the intermediate results of the W parameter of the embedding model.
            It saves the generated image and latent tensor to the 'W+' folder in the output directory.
            The intermediate results are saved with the format '{ref_name}_{step:04}.npy' and '{ref_name}_{step:04}.png'.
        """
        
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()

        intermediate_folder = os.path.join(self.opts.output_dir, 'W+', ref_name[0])
        os.makedirs(intermediate_folder, exist_ok=True)

        latent_path = os.path.join(intermediate_folder, f'{ref_name[0]}_{step:04}.npy')
        image_path = os.path.join(intermediate_folder, f'{ref_name[0]}_{step:04}.png')

        save_im.save(image_path)
        np.save(latent_path, save_latent)

    def save_FS_results(self, ref_name, gen_im: torch.Tensor, latent_in: torch.Tensor, latent_F: torch.Tensor) -> None:
        """
        Save the generated image and latent tensors to the output directory for the FS results.
        Args:
            ref_name (str): The reference name for the saved image and latent tensors.
            gen_im (torch.Tensor): The generated image tensor.
            latent_in (torch.Tensor): The input latent tensor.
            latent_F (torch.Tensor): The latent F tensor.
        Returns:
            None
        
        Description:
            Saves the generated image and latent tensors to the output directory for the FS results. 
        """
        
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))

        output_dir = os.path.join(self.opts.output_dir, 'FS')
        os.makedirs(output_dir, exist_ok=True)

        latent_path = os.path.join(output_dir, f'{ref_name[0]}.npz')
        image_path = os.path.join(output_dir, f'{ref_name[0]}.png')

        save_im.save(image_path)
        np.savez(latent_path, latent_in=latent_in.detach().cpu().numpy(),
                 latent_F=latent_F.detach().cpu().numpy())

    def set_seed(self):
        if self.opt.seed:
            torch.manual_seed(self.opt.seed)
            torch.cuda.manual_seed(self.opt.seed)
            torch.backends.cudnn.deterministic = True
