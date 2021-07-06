from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from loss import contrastive_loss
from models import Conv4
# from feature_bank import FeatureBank
import logging
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
def main():

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-bs','--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-li','--logging-interval', type=int, default=2 ,
                        help='how often to print loss, every nth')
    parser.add_argument('-sm','--save-model', type=bool, default=True ,
                        help='should model be saved')
    parser.add_argument('-e','--epochs', type=int, default=200 ,
                        help='how many epochs to run')
    parser.add_argument('-ns','--num_neg_samples', type=int, default=25 ,
                        help='how many negative samples to use')
    parser.add_argument('-wd',"--weight-decay",type=float,default=1e-4,
                        help=" weight decay for adam")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(filename='cifarpixelAsmaa_bs{}_ns{}.log'.format(args.batch_size, args.num_neg_samples), level=logging.DEBUG)                
    print(torch.cuda.is_available())
    print(device)
    # device = torch.device("cpu")
    

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
        CPCGridMaker((8,8))
        ])
    cifar_train = datasets.CIFAR10('./data', train=True, download=True,
                       transform=transform)
    cifar_test = datasets.CIFAR10('./data', train=False,
                       transform=transform)
    
    K=3
    num_neg_sample = args.num_neg_samples
    latent_size = 512
    
    train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(cifar_test)

    model = Conv4(img_channels=3,K=K,latent_size=latent_size).to(device)
    model = model.double()
    optimizer = optim.Adam(model.parameters(),weight_decay=args.weight_decay)
    
    for e in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_samples = 0
        for batch_idx,(data,_) in enumerate(train_loader):
            cur_batch = data.shape[0]
            data = data.to(device).double()
            loss = torch.tensor(0.0).to(device)
            num_samples += data.shape[0]
            grid_shape,img_shape = data.shape[:3],data.shape[3:]
            grid_shape_x = grid_shape[-1]
            output = model(data.view(-1,*img_shape))
            output = output.view(*grid_shape,-1)
            for t in range(grid_shape_x-K):
                for c in range(grid_shape_x):        
                    # print(output.shape)
                    enc_grid = output[:,:t+1,:,:].view(cur_batch,-1,latent_size)
                    enc_grid = enc_grid[:,:-(grid_shape_x-c-1) if (c <grid_shape_x-1) else grid_shape_x,:]
                    ar_out,_ = model.auto_regressive(enc_grid)
                    ar_out = ar_out[:,-1,:]
                    targets = output[:,t+1:t+K+1,c,:]
                    for k in range(K):
                        pos_sample = targets[:,k,:] 
                        neg_sample_idx = np.random.choice(grid_shape_x**2,num_neg_sample,replace=True)
                        neg_samples = output.view(cur_batch,-1,latent_size)[:,neg_sample_idx,:]
                        loss += contrastive_loss(pos_sample,neg_samples,model.W[k],ar_out,norm=True)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % args.logging_interval ==0:
                print("average Loss is {:.4f}, batch_idx is {}/{}".format(loss.item()/data.shape[0],batch_idx,len(train_loader)))
                logging.debug("average Loss is {:.4f}, batch_idx is {}/{}".format(loss.item()/data.shape[0],batch_idx,len(train_loader)))
        print("Loss is {}, epoch is {}".format(total_loss/num_samples,e))
        logging.debug("Loss is {}, epoch is {}".format(total_loss/num_samples,e))
        if args.save_model:
            if e%2 == 0:
                torch.save({
                    "model":model.state_dict(),
                    "opt":optimizer.state_dict()
                },"cifarpixelAsmaa_epoch{}_bs{}_ns{}.pt".format(e, args.batch_size, args.num_neg_samples))
                        
if __name__ == '__main__':
    main()