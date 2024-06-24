import sys
sys.path.insert(0, './')
sys.path.insert(0, './models/p3m_matting/core')


import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.ModelBase import Model
from network import build_model


DEFAULT_CHECKPOINT = "pretrained_models/P3M-Net_ViTAE-S_trained_on_P3M-10k.pth"
ARCH = 'vitae'
INFER_SIZE = 1024


class human_matt_model(Model):
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="cuda"):
        self.name = "HumanMatting"
        
        self.device = device
        
        # build model
        self.model = build_model(ARCH, pretrained=False)
        
        # load ckpt
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['state_dict'], strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.pil_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        
        self.NORMALIZE = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def inference(self, img, target_color=255):
        h, w = img.shape[-2:]
        """
        Input:
        imgs (torch tensor) (RGB): b x c x h x w

        Output:
        Tuple (
            seg_results (torch.Tensor): b x c x h x w
            seg_logits (torch.Tensor): b x h x w
        )
        """

        img = img.float().to(self.device) / 255
        img_norm = self.NORMALIZE(img)

        rh, rw = None, None
        if w >= h:
            rh = INFER_SIZE
            rw = int(w / h * INFER_SIZE)
        else:
            rw = INFER_SIZE
            rh = int(h / w * INFER_SIZE)
        rh = rh - rh % 64
        rw = rw - rw % 64
        
        input_tensor = F.interpolate(img_norm, size=(rh, rw), mode='bilinear')
        
        with torch.no_grad():
            _, _, pred_fusion = self.model(input_tensor)[:3]

        # output segment
        pred_fusion = F.interpolate(pred_fusion, size=(h, w), mode='bilinear')
    
        pred_fusion_np = pred_fusion[:, 0].data.cpu().numpy()
        img = img.data.cpu().numpy()
        
        pred_fusion_np = np.stack([pred_fusion_np, pred_fusion_np, pred_fusion_np], axis=1)
    
        base = np.ones_like(img) * target_color / 255
        distance = img - base
        new_img = (base + distance * pred_fusion_np) * 255
        new_img = new_img.astype(np.uint8)
        
        return new_img, pred_fusion[:, 0]
        
        
if __name__ == "__main__":
    path = "input/face/werewolf.png"
    image = Image.open(path)
    np_image = np.array(image.convert('RGB'))
    torch_img = torch.tensor(np_image).permute(2, 0, 1).unsqueeze(0)

    model = human_matt_model()

    results, __ = model.inference(torch_img)

    vis_img = Image.fromarray(results.transpose((0, 2, 3, 1))[0])
    vis_img.save(os.path.join("test_output", "backrem_" + os.path.basename(path)))
