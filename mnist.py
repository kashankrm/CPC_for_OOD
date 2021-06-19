from __future__ import print_function
import argparse
from datetime import datetime
from math import inf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
from torch.utils.tensorboard import SummaryWriter
import atexit

from torchvision.transforms.transforms import Resize
from matplotlib import pyplot as plt
from functools import reduce
import numpy as np
from loss import contrastive_loss
class TensorboardWriter():
    def __init__(self,logdir="tensorboard"):
        self.writer = SummaryWriter()
        self.data ={}
        self.delayed_scalers = {}
        atexit.register(self.cleanup)
    def register_delayed_scalars(self,name,cond):
        self.delayed_scalers[name] = cond

    def add_scalars_delayed(self,name,data,epoch):
        if name not in self.delayed_scalers:
            raise ValueError("delayed scaler named '{}' not found!".format(name))
        if name not in self.data:
            self.data[name] = {}
        if epoch not in self.data[name]:
            self.data[name][epoch] = {}
        for key,val in data.items():
            self.data[name][epoch][key] = val
        if set(self.delayed_scalers[name]) == set(self.data[name][epoch].keys()):
            self.add_scalars("delayed_"+name,self.data[name][epoch],epoch)
        
    def add_scalars(self,name,data,epoch):
        self.writer.add_scalars(name,data,epoch)
        if name in self.delayed_scalers:
            self.add_scalars_delayed(name,data,epoch)
    def cleanup(self):
        pass
class Net(nn.Module):
    def __init__(self,hidden_size=100,K=2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.auto_regressive = nn.GRU(1024,hidden_size,1)
        self.W = nn.ModuleList([nn.Linear(hidden_size,1024,bias=False) for i in range(K)] )
        
        self.K= K
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # # x = self.dropout2(x)
        # x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        # return output
class Pruner:
    def __init__(self,model,config):
        self.model = model
        self.strategy = config.get("strategy","OneShotMagnitudePrune") 
        ignore_first_layer=config.get("ignore_first_layer",True)
        ignore_last_layer=config.get("ignore_last_layer",True)
        if "masking_percentage" in config:
            if isinstance(config["masking_percentage"],int):
                self.linear_p = config["masking_percentage"]
                self.conv_p = config["masking_percentage"]

            elif isinstance(config["masking_percentage"],dict):
                self.linear_p = config["masking_percentage"].get("linear_p",0.2)
                self.conv_p = config["masking_percentage"].get("conv_p",0.1)
        if not hasattr(self,"linear_p"):
            self.linear_p = 0.2
            self.conv_p = 0.1
        
        self.ignore_first_layer = ignore_first_layer
        self.ignore_last_layer = ignore_last_layer
        self.calculate_masks(model)
        self.apply_masks(model)

    def calculate_masks(self,model):
        masks = {}
        module_list = list(model.named_modules())
        for name,module in module_list:
            if (name == module_list[-1][0] and self.ignore_last_layer) or (name == module_list[0][0] and self.ignore_first_layer):
                continue
            if any([isinstance(module,cl) for cl in [nn.Conv2d,nn.Linear]]):
                if isinstance(module,nn.Conv2d):
                    mask = self.magnitude_pruning(module.weight.data,p=self.conv_p).bool()    
                else:
                    mask = self.magnitude_pruning(module.weight.data,p=self.linear_p).bool()
                
                masks[name] = mask
        self.masks = masks
    def update_masks(self,model):
        module_list = model.named_modules()
        for name,module in module_list:
            if hasattr(module,"mask"):
                module.mask = self.masks[name]

    def apply_masks(self,model):
        def masking_forward_hook(self,_):
            self.weight = self.original_weight * self.mask
        module_list = list(model.named_modules())
        for name,module in module_list:
            if name == module_list[-1][0]:
                continue
            if any([isinstance(module,cl) for cl in [nn.Conv2d,nn.Linear]]):
                
                mask = self.masks[name]
                module.register_parameter("original_weight",module.weight)
                del module.weight
                
                module.register_buffer("mask",mask)
                
                module.register_forward_pre_hook( masking_forward_hook)
        
    def magnitude_pruning(self,tn,p=0.6):
        shape=None
        with torch.no_grad():
            shape = tn.shape
            tv = tn.view(-1)
            tv = torch.abs(tv)
            sorting = torch.argsort(tv)
            tn_size = len(sorting)
            mask = torch.ones_like(sorting)
            mask[sorting[:int(tn_size*p)]] = 0
            return torch.reshape(mask,shape)

    



def train(args, model, device, train_loader, optimizer, epoch):
    
    model.train()
    total_loss = 0.0
    total_correct = 0
    

    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = F.nll_loss(output, target)
        total_loss += loss.item()
        total_correct +=(output.argmax(dim=1) == target).sum()
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    
    return total_loss/len(train_loader.dataset), total_correct/len(train_loader.dataset)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)
class CPCGridMaker:
    def __init__(self,grid_shape):
        self.grid_shape = grid_shape
        
    def __call__(self,sample):
        
        sample = torch.squeeze(sample)
        grid_x,grid_y = self.grid_shape
        h, w = sample.shape
        out_shape_y,out_shape_x = h//(grid_y//2) -1, w//(grid_x//2) -1
        out = torch.zeros((out_shape_y,out_shape_x,grid_y,grid_x))
        for i in range(out_shape_y):
            for j in range(out_shape_x):
                out[i,j,:,:] = sample[i*(grid_y//2):i*(grid_y//2)+grid_y,j*(grid_x//2):j*(grid_x//2)+grid_x]
        return out
def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        CPCGridMaker((8,8))
        ])
    minist_train = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    minist_test = datasets.MNIST('./data', train=False,
                       transform=transform)
    grid_shape_x = 6
    K=2
    num_neg_sample = 10
    latent_size = 1024
    email_sara_mila_lo = False
    epochs = 10


    train_loader = torch.utils.data.DataLoader(minist_train,batch_size=8)
    test_loader = torch.utils.data.DataLoader(minist_test)

    model = Net(K=K).to(device)
    optimizer = optim.Adam(model.parameters())
    
    for e in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_idx,(data,_) in enumerate(train_loader):
            data = data.to(device)
            loss = torch.tensor(0.0).to(device)
            grid_shape,img_shape = data.shape[:3],data.shape[3:]
            output = model(torch.unsqueeze(data.view(-1,*img_shape),1))
            output = output.view(*grid_shape,-1)
            
            for c in range(grid_shape_x):
                cl = output[:,:,c,:]
                for t in range(grid_shape_x-K):
                    
                    inp = cl[:,:t+1,:]
                    

                    ar_out,hn = model.auto_regressive(inp)
                    ar_out = ar_out[:,-1,:]
                    targets = cl[:,t:t+K,:]
                    for k in range(K):
                        pos_sample = targets[:,k,:] 

                        neg_idx = [i for i in range(grid_shape_x) if i!=c]
                        
                        if email_sara_mila_lo:
                            ncl = output[:,:,neg_idx,:]
                            total_neg_sample = ncl.shape[1]*ncl.shape[2]
                            neg_samples_arr = ncl.view(-1,total_neg_sample,latent_size)
                        else:
                            ncl = output[:,k+t,neg_idx,:]
                            total_neg_sample = ncl.shape[1]
                            neg_samples_arr = ncl

                        neg_sample_idx = np.random.choice(total_neg_sample,num_neg_sample,replace=True)
                        neg_samples = neg_samples_arr[:,neg_sample_idx,:]
                        loss += contrastive_loss(pos_sample,neg_samples,model.W[k],ar_out,norm=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 ==0:
                print("immediate Loss is {}, batch_idx is {}/{}".format(loss.item(),batch_idx,len(train_loader)))
        print("Loss is {}, epoch is {}".format(total_loss,e))
                        

    

    
    
    

if __name__ == '__main__':
    main()