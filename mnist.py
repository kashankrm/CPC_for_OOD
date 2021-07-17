from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from loss import contrastive_loss
from models_mnist import Conv4
from CPCGridMaker import CPCGridMaker
import logging


def main():

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-bs','--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-li','--logging-interval', type=int, default=10,
                        help='how often to print loss, every nth')
    parser.add_argument('-sm','--save-model', type=bool, default=True ,
                        help='should model be saved')
    parser.add_argument('-e','--epochs', type=int, default=100 ,
                        help='how many epochs to run')
    parser.add_argument('-ns','--num_neg_samples', type=int, default=10 ,
                        help='how many negative samples to use')
    parser.add_argument('-wd',"--weight-decay",type=float,default=1e-5,
                        help=" weight decay for adam")
    parser.add_argument('-k','--K', type=int, default=2 ,
                        help='how many steps to predict')
    parser.add_argument('-rm',"--resume-model",type=str,help="path to ckpt to resume model")
    parser.add_argument('-cs',"--crop-size",type=int,default=8,help="crop size")
                           
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    args = parser.parse_args()

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        CPCGridMaker((args.crop_size,args.crop_size))
        ])

    minist_train = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    minist_test = datasets.MNIST('./data', train=False,
                       transform=transform)
    
    K=args.K
    num_neg_sample = args.num_neg_samples
    latent_size = 512
    train_loader = torch.utils.data.DataLoader(minist_train,batch_size=args.batch_size)
    model = Conv4(K=K,latent_size=latent_size).to(device)
    model = model.double()
    optimizer = optim.Adam(model.parameters(),weight_decay=args.weight_decay)

    if args.resume_model != None:
        ckpt = torch.load(args.resume_model)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
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
            grid_size = grid_shape[-1]
            output = model(torch.unsqueeze(data.view(-1,*img_shape),1))
            output = output.view(*grid_shape,-1)
            for t in range(grid_size-K):
                for c in range(grid_size):        
                    enc_grid = output[:,:t+1,:,:].view(cur_batch,-1,latent_size)
                    enc_grid = enc_grid[:,:-(grid_size-c-1) if (c <grid_size-1) else grid_size,:]
                    ar_out,_ = model.auto_regressive(enc_grid)
                    ar_out = ar_out[:,-1,:]
                    targets = output[:,t+1:t+K+1,c,:]
                    for k in range(K):
                        pos_sample = targets[:,k,:] 
                        neg_sample_idx = np.random.choice(grid_size**2,num_neg_sample,replace=True)
                        neg_samples = output.view(cur_batch,-1,latent_size)[:,neg_sample_idx,:]
                        loss += contrastive_loss(pos_sample,neg_samples,model.W[k],ar_out,norm=False)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % args.logging_interval ==0:
                print("average Loss is {:.4f}, batch_idx is {}/{}".format(loss.item()/data.shape[0],batch_idx,len(train_loader)))
                logging.debug("average Loss is {:.4f}, batch_idx is {}/{}".format(loss.item()/data.shape[0],batch_idx,len(train_loader)))
        print("Loss is {}, epoch is {}".format(total_loss/num_samples,e))
        logging.debug("Loss is {}, epoch is {}".format(total_loss/num_samples,e))
        if args.save_model:
            if (e+1) % 10 == 0:
                torch.save({
                    "model":model.state_dict(),
                    "opt":optimizer.state_dict(),
                    "K" : args.K,
                    "epoch" : e,
                    "latent_size": latent_size,
                    "crop_size":args.crop_size
                    },"mnist_epoch{}_ns{}_k{}.pt".format(e,args.num_neg_samples,args.K))
                   

if __name__ == '__main__':
    main()