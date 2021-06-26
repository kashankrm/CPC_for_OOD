from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from loss import contrastive_loss
from models import Conv4,LinClassifier
from feature_bank import FeatureBank
from yellowbrick.text import TSNEVisualizer
from yellowbrick.features import PCA

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
    

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        CPCGridMaker((14,14))
        ])
    minist_train = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    minist_test = datasets.MNIST('./data', train=False,
                       transform=transform)
    grid_shape_x = 3
    K=args.K
    num_neg_sample = args.num_neg_samples
    latent_size = 128
    email_sara_mila_lo = False
    


    # train_loader = torch.utils.data.DataLoader(minist_train,batch_size=args.batch_size,num_workers=4)
    test_loader = torch.utils.data.DataLoader(minist_test)

    model = LinClassifier("cpc_models/mnist_epoch27.pt").to(device)
    
    model = model.double()
    out_data ={}
    
    for batch_idx,(data,target) in enumerate(test_loader):
        cur_batch = data.shape[0]
        data = data.to(device).double()
        grid_shape,img_shape = data.shape[:3],data.shape[3:]
        output = model.feat(torch.unsqueeze(data.view(-1,*img_shape),1))
        output = output.view(*grid_shape,-1)
        for i,t in enumerate(target):
            k = t.item()
            out_data[k] = out_data.get(k,[])
            out_data[k].append(output.mean(dim=[1,2])[i].detach().cpu().numpy()) 
    X = []
    y = []
    for key in out_data:
        X = X + out_data[key]
        y = y + [key]*len(out_data[key])
        out_data[key] = np.stack(out_data[key],axis=0)
    X = np.stack(X,axis=0)
    y = np.stack(y,axis=0)
    pca = PCA(scale=True)
    pca.fit_transform(X,y)
    pca.show()
    
    
    
    print("")                   

    

    
    
    

if __name__ == '__main__':
    main()