import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl 
import os
class CPCGridMaker:
    def __init__(self,grid_shape):
        self.grid_shape = grid_shape
        
    def __call__(self,sample):
        
        sample = torch.squeeze(sample)
        grid_x,grid_y = self.grid_shape
        d, h, w = sample.shape if len(sample.shape)>2 else (0,*sample.shape)
        out_shape_y,out_shape_x = h//(grid_y//2) -1, w//(grid_x//2) -1
        
        if d ==0:
            out = torch.zeros((out_shape_y,out_shape_x,grid_y,grid_x))
        else:
            out = torch.zeros((out_shape_y,out_shape_x,d,grid_y,grid_x,))
        for i in range(out_shape_y):
            for j in range(out_shape_x):
                if d ==0:
                    out[i,j,:,:] = sample[i*(grid_y//2):i*(grid_y//2)+grid_y,j*(grid_x//2):j*(grid_x//2)+grid_x]
                else:
                    out[i,j,:,:,:] = sample[:,i*(grid_y//2):i*(grid_y//2)+grid_y,j*(grid_x//2):j*(grid_x//2)+grid_x]
        return out

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage):
        # transforms for images
        transform=transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    CPCGridMaker((14,14))])
        
        # prepare transforms standard to MNIST
        self.mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
        self.mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size,num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size,num_workers=4)
