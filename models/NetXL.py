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
from models.ModelBase import Model


class NetXL(nn.Module, Model):
    def __init__(self, opts: dict, device):
        super(NetXL, self).__init__()

        self.name = "StyleGanXL"
        self.opts = opts
        self.device = device
        self.load_weights()

    def inference(self, latent):
        return self.generator.synthesis(latent, noise_mode="const")
    
    def load_weights(self):
        print('Loading networks from "%s"...' % self.opts.ckpt)
        
        with dnnlib.util.open_url(self.opts.ckpt) as f:
            loaded_network = legacy.load_network_pkl(f)

        # print(loaded_network.keys())
        self.generator = loaded_network['G_ema']
        self.generator = self.generator.eval().requires_grad_(False).to(self.device)
        self.generator.eval()

        for param in self.generator.parameters():
            param.requires_grad = False
        
        if self.opts.size == 1024:
            self.default_latent = torch.zeros((1, 22, 512), dtype=torch.float32)
        elif self.opts.size == 512:
            self.default_latent = torch.zeros((1, 20, 512), dtype=torch.float32)
        else:
            raise NotImplementedError(f"Image size ({self.opts.size}) has not been implemented in StyleGanXL")
            

if __name__ == "__main__":
    class opt:
        def __init__(self):
            self.size = 1024
            self.latent = 512
            self.channel_multiplier = 2
            self.n_mlp = 8
            self.ckpt = 'pretrained_models/ffhq1024xl.pkl'
    
    net = NetXL(opt(), 'cuda')
        
    latent = net.default_latent

    # Average generation time 0.08938145637512207
    gen_im = net.inference(latent)

