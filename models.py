import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.modules.linear import Linear

class Conv4(nn.Module):
    def __init__(self,img_channels=1,hidden_size=256,K=2,latent_size=512):
        super().__init__()
        self.latent_size = latent_size
        self.feature = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.auto_regressive = nn.GRU(latent_size,hidden_size,2)
        self.W = nn.ModuleList([nn.Linear(hidden_size,latent_size,bias=False) for i in range(K)] )
        self.W = nn.ModuleList([nn.Sequential(nn.Linear(in_features=hidden_size, out_features=hidden_size//2, bias=True),
        nn.Dropout(),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size//2, out_features=latent_size, bias=True)) for i in range(K) ] )     
        self.K= K
        
    def forward(self, x):
        x = self.feature(x)
        return x

class Linear_Layer(nn.Module):
    def __init__(self,input_size = 25088, classes = 10):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(input_size, input_size//2, bias=True),
        nn.Dropout(),
        nn.ReLU(inplace=False),
        nn.Linear(input_size//2, 512, bias=True),
        nn.Dropout(),
        nn.ReLU(inplace=False),
        nn.Linear(512, classes, bias=True))
    
    def forward(self, x):
        x = self.feature(x)
        return x