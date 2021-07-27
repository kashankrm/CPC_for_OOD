from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from loss import contrastive_loss
from utils import CPCGridMaker
from models_mnist import Conv4, LinClassifier

from yellowbrick.text import TSNEVisualizer
from matplotlib import pyplot as plt


def main():

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

    # device = torch.device("cpu")
    
    data_points = 500
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        CPCGridMaker((8,8))
        ])
    transform_omniglot = transforms.Compose([
        transforms.Resize((28,28)),
        lambda x:transforms.functional.invert(x),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        CPCGridMaker((8, 8))
    ])

    
    
    omniglot_test = datasets.Omniglot('./data',background=False, download=True,
                       transform=transform_omniglot)
    mnist_test = datasets.MNIST('./data', train=False,
                       transform=transform,download=True)
    mnist_subset = np.random.choice(np.arange(0,len(mnist_test)),data_points)
    omniglot_subset = np.random.choice(np.arange(0,len(omniglot_test)),data_points)
    omniglot_test = torch.utils.data.Subset(omniglot_test, omniglot_subset)
    mnist_test = torch.utils.data.Subset(mnist_test, mnist_subset)
    
    grid_shape_x = 6
    K=args.K
    num_neg_sample = args.num_neg_samples
    latent_size = 128
    email_sara_mila_lo = False
    


    # train_loader = torch.utils.data.DataLoader(minist_train,batch_size=args.batch_size,num_workers=4)
    mnist_loader = torch.utils.data.DataLoader(mnist_test,batch_size=256,num_workers=0)
    omniglot_loader = torch.utils.data.DataLoader(omniglot_test,batch_size=256,num_workers=0)

    model = LinClassifier("/home/kashankarimudin/asmaa_google_pt/mnist_epoch89_ns30_k3.pt").to(device)
    
    model = model.double()
    mnist_embed = []
    
    for batch_idx,(data,target) in enumerate(mnist_loader):
        
        cur_batch = data.shape[0]
        data = data.to(device).double()
        grid_shape,img_shape = data.shape[:3],data.shape[3:]
        output = model.feat(torch.unsqueeze(data.view(-1,*img_shape),1))
        output = output.view(*grid_shape,-1)
        mnist_embed.append(output.view(cur_batch,-1).detach().cpu().numpy()) 
        
        
            
    omniglot_embed = []
    for batch_idx,(data,target) in enumerate(omniglot_loader):
        cur_batch = data.shape[0]
        data = data.to(device).double()
        grid_shape,img_shape = data.shape[:3],data.shape[3:]
        output = model.feat(torch.unsqueeze(data.view(-1,*img_shape),1))
        output = output.view(*grid_shape,-1)
        
        omniglot_embed.append(output.view(cur_batch,-1).detach().cpu().numpy()) 
    mnist_embed = np.concatenate(mnist_embed,axis=0)
    omniglot_embed = np.concatenate(omniglot_embed,axis=0)
    X = np.concatenate([mnist_embed,omniglot_embed],axis=0)
    y = np.array([*["minst"]*data_points,*["omniglot"]*data_points])

    
    

    tsne = TSNEVisualizer(alpha=0.8)
    tsne.fit(X,y)
    tsne.finalize()
    plt.savefig("mnist_vs_omniglot.png")
    # tsne.show()
    
    
    
    print("")                   

    

    
    
    

if __name__ == '__main__':
    main()