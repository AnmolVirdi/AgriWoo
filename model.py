#Importing Pytorch, OpenCV, Albemntations
import torch
import cv2
import albumentations as A
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torchmetrics import (Accuracy, ConfusionMatrix,)
import categories
from PIL import Image
import numpy
from torchinfo import summary
import numpy as np
import csv

class PlantDiseasesDataset(Dataset):
    def __init__(self, path, transform=None):
        self.files = glob(path, recursive=True)
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = A.Compose([
                A.Normalize(),
                ToTensorV2(),
            ])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        item = self.files[idx]
        
        fruit = None
        label = None
        
        for fruit_i in categories.category.keys():
            if fruit_i in item:
                fruit = categories.category[fruit_i]
                break
                
        for dis in fruit[1].keys():
            if dis in item:
                label = fruit[1][dis]
                break

        img = cv2.imread(item)
        
        fruit_channel = torch.ones(1, img.shape[0], img.shape[1])
        fruit_channel *= fruit[0] / 13        
        
        img = self.transform(image=img)['image']
        
        img = torch.vstack([img, fruit_channel])
        
        if label is None or fruit is None:
            print(label, fruit)
        
        return img, fruit[0], label

class NeuralNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(NeuralNet, self).__init__()
        self.in_channels = in_channels
        self.input_layer = nn.Sequential(
#                 nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, 3, 3, 1, 1)
        )
        
        self.efficientnet = models.efficientnet_b7(pretrained=True)
        
        self.last = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        
    def forward(self, x,):
        if self.in_channels != 3:
            x = self.input_layer(x)
            
        x = self.efficientnet(x)
#         x = torch.cat([x, fruit])
        x = self.last(x)
        
        return x


def predict(img, fruit):
    """
    1, 4, 224, 224
    """
    
    _fruit = categories.category[fruit][0]
    fruit_channel = torch.ones(1, img.shape[0], img.shape[1])
    fruit_channel *= _fruit / 13   
    
    transform = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])
    
    img = transform(image=img)['image']
    img = torch.vstack([img, fruit_channel])
    

    model = NeuralNet(4, 38)
    # print(summary(model, (1,4,224,224)))
    model.load_state_dict(torch.load('model_checkpoint_1.pt')['model'])
    
    model = model.cpu()
    
    # with model.eval():
    pred = torch.argmax(torch.softmax(model(img.unsqueeze(dim=0)), dim=1))

    return categories.idx2category(int(pred), fruit), int(pred)


def main():
    import cv2
    import matplotlib.pyplot as plt
    
    img = cv2.imread('./1.jpg')
    plt.imshow(img)
    plt.show()
    
    # print(predict(img, 'Apple'))
    disease, prediction=predict(img, 'Apple')
    print(disease)
    for i in categories.category.values():
        for j in i[1].values():
            
            # if j==prediction and j not in (3,4,6,10,14,16,18,20,23,24,25,28):
                
            if disease != "Healthy" and j == prediction:
                with open('disease.csv', 'r') as f:
                    mycsv = list(csv.reader(f))
                    print(mycsv[j][1])
                
            elif j==prediction:
                print("Healthy")
    
if __name__ == "__main__":
    main()
    


# img = cv2.imread('1.jpg')
# print(img)
# img = cv2.resize(img, (224, 224))

# img = torch.tensor(img).permute(2, 0, 1)
# # print(img)

# print(predict(img))