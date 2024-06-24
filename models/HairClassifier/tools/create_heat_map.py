import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"

import sys
sys.path.insert(0, "./")

import numpy as np

import torch
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import time

from mobilenetv3 import mobilenet_v3_large
from train import create_dataloader
from dataset import Split


@torch.no_grad()
def create_heatmap():
    net = mobilenet_v3_large(num_classes=6)
    net.load_state_dict(torch.load("weights/mobilenetv3_split/model-59.pth"))
    net = net.cuda()
    net.eval()
    
    test_dataloader = create_dataloader(split=Split.TEST, batch_size=1, shuffle=True)

    heatmap = np.zeros((6, 6))
    
    for batch_idx, (test_features, test_labels) in enumerate(test_dataloader):
        test_features = test_features.cuda()
        test_labels = test_labels.numpy()

        inference: torch.Tensor = net(test_features)

        predicted = inference.argmax(dim=1).detach().cpu().numpy()
        for i in range(predicted.shape[0]):
            heatmap[predicted[i], test_labels[i]] += 1

    print(heatmap)
    
    colors_list = ['#0099ff', '#33cc33']
    cmap = colors.ListedColormap(colors_list)
    
    plt.imshow(heatmap, cmap=cmap, vmin=0, vmax=100, extent=[0, 6, 0, 6])
    for i in range(6):
        for j in range(6):
            plt.annotate(str(heatmap[i, j]), xy=(j+0.5, i+0.5),
                         ha='center', va='center', color='white')
      
    # Add colorbar
    cbar = plt.colorbar(ticks=[0, 50, 100])
    cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])
      
    # Set plot title and axis labels
    plt.title("Customized heatmap with annotations")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
  
    plt.savefig('heatmap.png')


if __name__ == "__main__":
    create_heatmap()
    