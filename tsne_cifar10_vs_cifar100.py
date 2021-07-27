from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from loss import contrastive_loss
from utils import CPCGridMaker
from models_cifar10 import Conv4, LinClassifier

from yellowbrick.text import TSNEVisualizer
from matplotlib import pyplot as plt


def main(args,model_name):

    
    # device = torch.device("cpu")
    
    data_points = 200
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        CPCGridMaker((8,8))
        ])
    transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        CPCGridMaker((8, 8))
    ])

    
    
    cifar100_test = datasets.CIFAR100('./data',train=False, download=True,
                       transform=transform_cifar10)
    cifar10_test = datasets.CIFAR10('./data', train=False,
                       transform=transform,download=True)
    cifar10_subset = np.random.choice(np.arange(0,len(cifar10_test)),data_points)
    cifar100_subset = np.random.choice(np.arange(0,len(cifar100_test)),data_points)
    cifar100_test = torch.utils.data.Subset(cifar100_test, cifar100_subset)
    cifar10_test = torch.utils.data.Subset(cifar10_test, cifar10_subset)
    
    
    K=args.K
    


    # train_loader = torch.utils.data.DataLoader(minist_train,batch_size=args.batch_size,num_workers=4)
    cifar10_loader = torch.utils.data.DataLoader(cifar10_test,batch_size=256,num_workers=0)
    cifar100_loader = torch.utils.data.DataLoader(cifar100_test,batch_size=256,num_workers=0)
    # model_name = "cifar_resnet_epoch89_bs64_ns40_cs8_hs100_g3"
    model = LinClassifier(f"E:\\study\\sem_4\\dl_lab\\project\\cpc_models\\asmaa_pt\\{model_name}.pt").to(device)
    
    model = model.double()
    cifar10_embed = []
    
    for batch_idx,(data,target) in enumerate(cifar10_loader):
        
        cur_batch = data.shape[0]
        data = data.to(device).double()
        grid_shape,img_shape = data.shape[:3],data.shape[3:]
        output = model.feat(data.view(-1,*img_shape))
        output = output.view(*grid_shape,-1)
        cifar10_embed.append(output.view(cur_batch,-1).detach().cpu().numpy()) 
        
        
            
    cifar100_embed = []
    for batch_idx,(data,target) in enumerate(cifar100_loader):
        
        cur_batch = data.shape[0]
        data = data.to(device).double()
        grid_shape,img_shape = data.shape[:3],data.shape[3:]
        output = model.feat(data.view(-1,*img_shape))
        output = output.view(*grid_shape,-1)
        
        cifar100_embed.append(output.view(cur_batch,-1).detach().cpu().numpy()) 
    cifar10_embed = np.concatenate(cifar10_embed,axis=0)
    cifar100_embed = np.concatenate(cifar100_embed,axis=0)
    X = np.concatenate([cifar10_embed,cifar100_embed],axis=0)
    y = np.array([*["CIFAR-10"]*data_points,*["CIFAR-100"]*data_points])

    
    

    tsne = TSNEVisualizer(alpha=0.8)
    tsne.fit(X,y)
    tsne.finalize()
    plt.savefig(f"cifar10_vs_cifar100_dt{data_points}_{model_name}.png")
    plt.gcf().clear()
    # tsne.show()
    
    
    
    print("")                   

    

    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-bs','--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-li','--logging-interval', type=int, default=100 ,
                        help='how often to print loss, every nth')
    parser.add_argument('-sm','--save-model', type=bool, default=True ,
                        help='should model be saved')
    parser.add_argument('-e','--epochs', type=int, default=20 ,
                        help='how many epochs to run')
    parser.add_argument('-ns','--num_neg_samples', type=int, default=20 ,
                        help='how many negative samples to use')
    parser.add_argument('-wd',"--weight-decay",type=float,default=1e-5,
                        help=" weight decay for adam")
    parser.add_argument('-k','--K', type=int, default=2 ,
                        help='how many steps to predict')
    
                           
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    args = parser.parse_args()
    model_list = ["cifar_resnet_epoch149_bs64_ns40_cs8","cifar_epoch118_bs64_ns40_cs8_hs100_g3","cifar_sub_epoch10_bs50_ns30_cs8_hs100_g1"]
    for model_name in model_list:
        main(args,model_name)