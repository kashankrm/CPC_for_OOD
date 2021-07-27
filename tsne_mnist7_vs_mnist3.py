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

def main(args,model_name):
    # device = torch.device("cpu")
    
    data_points = 200
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        CPCGridMaker((8,8))
        ])
    # mnist3_test = datasets.EMNIST('./data',split="letters", train=False, download=True,
    #                    transform=transform)
    mnist_test = datasets.MNIST('./data', train=False,
                       transform=transform,download=True)
    included_classes = [0,1,2,3,4,5,6]
    subset7 = [i for i,v in enumerate(mnist_test.targets) if v in included_classes]
    mnist7_subset = np.random.choice(subset7,data_points)
    mnist3_subset = np.random.choice(list(set(range(len(mnist_test))) - set(subset7)),data_points)
    mnist3_test = torch.utils.data.Subset(mnist_test, mnist3_subset)
    mnist7_test = torch.utils.data.Subset(mnist_test, mnist7_subset)
    
    grid_shape_x = 6
    K=args.K
    num_neg_sample = args.num_neg_samples
    latent_size = 128
    email_sara_mila_lo = False
    


    # train_loader = torch.utils.data.DataLoader(minist_train,batch_size=args.batch_size,num_workers=4)
    mnist7_loader = torch.utils.data.DataLoader(mnist7_test,batch_size=256,num_workers=0)
    mnist3_loader = torch.utils.data.DataLoader(mnist3_test,batch_size=256,num_workers=0)

    model = LinClassifier(f"E:\\study\\sem_4\\dl_lab\\project\\cpc_models\\asmaa_pt\\{model_name}.pt").to(device)
    
    model = model.double()
    mnist7_embed = []
    
    for batch_idx,(data,target) in enumerate(mnist7_loader):
        
        cur_batch = data.shape[0]
        data = data.to(device).double()
        grid_shape,img_shape = data.shape[:3],data.shape[3:]
        output = model.feat(torch.unsqueeze(data.view(-1,*img_shape),1))
        output = output.view(*grid_shape,-1)
        mnist7_embed.append(output.view(cur_batch,-1).detach().cpu().numpy()) 
        
        
            
    mnist3_embed = []
    for batch_idx,(data,target) in enumerate(mnist3_loader):
        
        cur_batch = data.shape[0]
        data = data.to(device).double()
        grid_shape,img_shape = data.shape[:3],data.shape[3:]
        output = model.feat(torch.unsqueeze(data.view(-1,*img_shape),1))
        output = output.view(*grid_shape,-1)
        
        mnist3_embed.append(output.view(cur_batch,-1).detach().cpu().numpy()) 
    mnist7_embed = np.concatenate(mnist7_embed,axis=0)
    mnist3_embed = np.concatenate(mnist3_embed,axis=0)
    X = np.concatenate([mnist7_embed,mnist3_embed],axis=0)
    y = np.array([*["MNIST (7 classes)"]*data_points,*["MNIST (3 classes)"]*data_points])

    
    

    tsne = TSNEVisualizer(alpha=0.8)
    tsne.fit(X,y)
    tsne.finalize()
    plt.savefig(f"mnist7_vs_mnist3_{model_name}.png")
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
    
                           
    device = torch.device("cuda" if torch.cuda.is_available() and False  else "cpu")
    args = parser.parse_args()
    model_name = "mnist7_epoch89_ns30_k3"
    model_list = ["mnist_sub_epoch49_ns15_k2","mnist_sub_epoch59_ns15_k2"]
    for model_name in model_list:
        main(args,model_name)