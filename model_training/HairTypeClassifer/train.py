import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"

import sys
sys.path.insert(0, "./")

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset import HairTypeDataset, Split

from mobilenetv3 import mobilenet_v3_large

NUM_CLASSES = 6
NUM_EPOCHS = 60
BATCH_SIZE = 30

LOG_INTERVAL = 200

DATA_ROOT = "data/hair_only"
WEIGHTS_ROOT = "weights/mobilenetv3/"

LOSS = CrossEntropyLoss()


def create_dataloader(num_workers=2, shuffle=True, split=Split.ALL, batch_size=BATCH_SIZE):
    dataset = HairTypeDataset(DATA_ROOT, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataloader
    

def train(weight_pth=None, whole=False):
    if not whole:
        train_dataloader = create_dataloader(split=Split.TRAIN)
        test_dataloader = create_dataloader(split=Split.TEST)
    else:
        train_dataloader = create_dataloader(split=Split.ALL)

    cur_epoch = 0
    
    net = mobilenet_v3_large(num_classes=NUM_CLASSES)

    if weight_pth is not None:
        cur_epoch = int(weight_pth[weight_pth.index("-") + 1: weight_pth.index(".")]) + 1
        net = mobilenet_v3_large(num_classes=NUM_CLASSES)
        net.load_state_dict(torch.load(weight_pth))
        
    net = net.cuda()
    
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    lr_params = [
        (0, 0.01),
        (30, 0.001),
        (45, 0.0001),
    ]
    
    writer = SummaryWriter()
    
    for epoch in range(cur_epoch, NUM_EPOCHS):

        lr = 0
        for lr_set in lr_params:
            if epoch >= lr_set[0]:
                lr = lr_set[1]
            else:
                break

        for g in optimizer.param_groups:
            g['lr'] = lr
                
        total_correct = 0
        total = 0

        net.train()
        
        for batch_idx, (train_features, train_labels) in enumerate(train_dataloader):
            train_features = train_features.cuda()
            train_labels = train_labels.cuda()
            
            inference = net(train_features)
            
            predicted = inference.argmax(dim=1)
        
            for i in range(predicted.shape[0]):
                if predicted[i] == train_labels[i]:
                    total_correct += 1
                total += 1
            
            loss = LOSS(inference, train_labels)
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            optimizer.zero_grad()

        train_set_accuracy = total_correct / total
        print("Training set accuracy: {} on epoch {}".format(train_set_accuracy, epoch))

        if not whole:
            test_set_accuracy = get_test_accuracy(net, test_dataloader, epoch)
            writer.add_scalar("Accuracy/test", test_set_accuracy, epoch)
        
        writer.add_scalar("Accuracy/train", train_set_accuracy, epoch)
        writer.flush()
        
        os.makedirs(WEIGHTS_ROOT, exist_ok=True)
        epoch_save_dir = os.path.join(WEIGHTS_ROOT, f"model-{epoch}.pth")
        torch.save(net.state_dict(), epoch_save_dir)

        print("Saved model at {}".format(epoch_save_dir))


@torch.no_grad()
def get_test_accuracy(net, test_dataloader, epoch):
    total_correct = 0
    total = 0

    net.eval()
    
    for batch_idx, (test_features, test_labels) in enumerate(test_dataloader):
        test_features = test_features.cuda()
        test_labels = test_labels.cuda()
            
        inference: torch.Tensor = net(test_features)
            
        predicted = inference.argmax(dim=1)
        
        for i in range(predicted.shape[0]):
            if predicted[i] == test_labels[i]:
                total_correct += 1
            total += 1
            
    accuracy = total_correct / total
    print("Test set accuracy: {} on epoch {}".format(accuracy, epoch))
    
    return accuracy
        

if __name__ == "__main__":
    model_pth = os.path.join(WEIGHTS_ROOT, "model-45.pth")
    model_pth = None
    train(model_pth, whole=False)
    
        