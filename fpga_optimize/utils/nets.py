import torch
from torch import nn




class largeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
                # Block 1
                # 128x128
                nn.Conv2d(3,64,3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                # Block 2
                # 64x64
                nn.Conv2d(64,128,3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                # Block 3
                # 32x32
                nn.Conv2d(128,256,3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256,256,3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                # Block 4
                # 16x16
                nn.Conv2d(256,512,3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,512,3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                # 8x8
                # Block 5
                nn.Conv2d(512,512,3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,512,3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                #4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(8192,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,100),
            )
    def forward(self,x):
        x = self.features(x)
        x = self.flatten(x)
        z = self.classifier(x)
        return z


class giantNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
                # Block 1 128x128
                nn.Conv2d(3,64,3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                # Block 2 64x64
                nn.Conv2d(64,128,3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                # Block 3 32x32
                nn.Conv2d(128,256,3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256,256,3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256,256,3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256,256,3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                # Block 4 16x16
                nn.Conv2d(256,512,3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,512,3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,512,3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,512,3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                # Block 5 8x8
                nn.Conv2d(512,512,3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,512,3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,512,3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,512,3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2)
                # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(8192,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,100),
            )
    def forward(self,x):
        x = self.features(x)
        x = self.flatten(x)
        z = self.classifier(x)
        return z


class mediumNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
                # Block 1
                #128x128
                nn.Conv2d(3,96,7,padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                # Block 2
                #64x64
                nn.Conv2d(96,256,5,padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                # Block 3
                #32x32
                nn.Conv2d(256,384,3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                #16x16
                # Block 4
                nn.Conv2d(384,384,3,padding=0),
                nn.ReLU(inplace=True),
                # Block 5
                #14x14
                nn.Conv2d(384,256,3,padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(9216,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,100),
            )
    def forward(self,x):
        x = self.features(x)
        x = self.flatten(x)
        z = self.classifier(x)
        return z

class smallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
                # Block 1
                #128 128
                nn.Conv2d(3,128,7,padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),

                # Block 2
                #64x64
                nn.Conv2d(128,256,5,padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                # Block 3
                #32x32
                nn.Conv2d(256,256,3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                #16x16
                # Block 4
                nn.Conv2d(256,512,3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                #8x8
                nn.Conv2d(512,512,3,padding=0),
                #6x6
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4608,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,100),
            )
    def forward(self,x):
        x = self.features(x)
        x = self.flatten(x)
        z = self.classifier(x)
        return z

class testNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
        nn.Conv2d(1,2,5,padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
                nn.Linear(32,4)
                )
    def forward(self,x):
        x = self.features(x)
        x = self.flatten(x)
        z = self.classifier(x)
        return z


def printModelSize(model):
    params = sum(p.numel() for p in model.parameters())
    paramkb= params * 4 /(1024)
    print("Model size [kB] : %lf"%(paramkb))

