
import torch

from dataset import INFERENCE_TRANSFORM
from mobilenetv3 import mobilenet_v3_large
from train import NUM_CLASSES

class HairClassifier:
    
    def __init__(self, weight_pth):
        self.model = mobilenet_v3_large(num_classes=NUM_CLASSES)
        self.model.load_state_dict(torch.load(weight_pth))
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
        
    def predict(self, image):
        image = INFERENCE_TRANSFORM(image)
        image = image.to(self.model.device)
        return self.model(image)