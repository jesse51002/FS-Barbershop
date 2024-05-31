import sys
sys.path.insert(0,'./')
sys.path.insert(0,'./models/styleganxl')

import torch
from torch import nn
import legacy
import dnnlib
import numpy as np
import os
import time
import tqdm

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class Net(nn.Module):
    def __init__(self, opts):
        super(Net, self).__init__()
        self.opts = opts
        self.cal_layer_num()
        self.load_weights()
        # self.load_PCA_model()

    def load_weights(self):
        print('Loading networks from "%s"...' % self.opts.ckpt)
        
        device = self.opts.device
        with dnnlib.util.open_url(self.opts.ckpt) as f:
            loaded_network = legacy.load_network_pkl(f)

        # print(loaded_network.keys())
        self.generator = loaded_network['G_ema']
        self.generator = self.generator.eval().requires_grad_(False).to(device)
    
        self.latent_avg = torch.ones((1, 20, 512))

        self.generator.eval()

    def create_img_from_latent(self, latent):
        return self.generator.synthesis(latent, noise_mode="const")

    def build_PCA_model(self, PCA_path):
        from utils.PCA_utils import IPCAEstimator
        transformer = IPCAEstimator(512)

        batch_size = 10

        norm = PixelNorm()
        # self.generator.cpu()
        
        for i in range(0, 1000000, batch_size):
            print(f"{i} / 1000000")
            with torch.no_grad():
                latent = torch.randn((batch_size, 20, 512), dtype=torch.float32).to(self.opts.device)
                pulse_space = torch.nn.LeakyReLU(5)(norm(latent)).detach().cpu().numpy()
            X_mean = pulse_space.mean(0)
            transformer.fit_partial(pulse_space - X_mean)

        X_comp, X_stdev, X_var_ratio = transformer.get_components()
        np.savez(PCA_path, X_mean=X_mean, X_comp=X_comp, X_stdev=X_stdev, X_var_ratio=X_var_ratio)

        self.generator.to(self.opts.device)


    def load_PCA_model(self):
        device = self.opts.device

        PCA_path = self.opts.ckpt[:-3] + '_PCA.npz'

        if not os.path.isfile(PCA_path):
            self.build_PCA_model(PCA_path)

        PCA_model = np.load(PCA_path)
        self.X_mean = torch.from_numpy(PCA_model['X_mean']).float().to(device)
        self.X_comp = torch.from_numpy(PCA_model['X_comp']).float().to(device)
        self.X_stdev = torch.from_numpy(PCA_model['X_stdev']).float().to(device)



    # def make_noise(self):
    #     noises_single = self.generator.make_noise()
    #     noises = []
    #     for noise in noises_single:
    #         noises.append(noise.repeat(1, 1, 1, 1).normal_())
    #
    #     return noises

    def cal_layer_num(self):
        if self.opts.size == 1024:
            self.layer_num = 18
        elif self.opts.size == 512:
            self.layer_num = 16
        elif self.opts.size == 256:
            self.layer_num = 14

        self.S_index = self.layer_num - 11

        return


    def cal_p_norm_loss(self, latent_in):
        latent_p_norm = (torch.nn.LeakyReLU(negative_slope=5)(latent_in) - self.X_mean).bmm(
            self.X_comp.T.unsqueeze(0)) / self.X_stdev
        p_norm_loss = self.opts.p_norm_lambda * (latent_p_norm.pow(2).mean())
        return p_norm_loss


    def cal_l_F(self, latent_F, F_init):
        return self.opts.l_F_lambda * (latent_F - F_init).pow(2).mean()


if __name__ == "__main__":
    class opt:
        def __init__(self):
            self.size = 1024
            self.latent = 512
            self.channel_multiplier = 2
            self.n_mlp = 8
            self.ckpt = 'pretrained_models/ffhq512xl.pkl'
            self.device = 'cuda'
    
    net = Net(opt())
        
    latent = torch.randn((1, 20, 512), dtype=torch.float32).clone().cuda()

    # Average generation time 0.05306898355484009
    
    net.create_img_from_latent(latent)

    optimizer_align, latent_align_1 = self.setup_align_optimizer(cur_net, latent_W_path_1, device=cur_device)

    cur_target_mask = target_mask.to(cur_device)

    align_steps = 50

    def create_down_seg(cur_seg, net, latent_in):
        # Fill in
        print()
            
    pbar = tqdm(range(align_steps), desc='Align Step 1', leave=False, disable=False)
    for step in pbar:
        optimizer_align.zero_grad()
        latent_in = torch.cat([latent_align_1[:, :6, :], latent_1[:, 6:, :]], dim=1)
        down_seg, _ = create_down_seg(cur_seg, net, latent_in)
                
        loss_dict = {}
                
        # Cross Entropy Loss
        ce_loss = cur_loss_builder.cross_entropy_loss(down_seg, cur_target_mask)
        loss_dict["ce_loss"] = ce_loss.item()
        loss = ce_loss
        # print(loss_dict)
                
        loss.backward()
        optimizer_align.step()
    
    intermediate_align, _ = cur_net.generator([latent_in], input_is_latent=True, return_latents=False,
                                                       start_layer=0, end_layer=3)
    intermediate_align = intermediate_align.clone().detach()

    
    print("Finsihed image gen in", (time.time() - start) / 20)

