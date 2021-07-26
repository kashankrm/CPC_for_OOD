from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from loss import contrastive_loss
from models_mnist import Conv4
from utils import CPCGridMaker, train, get_mnist_loader
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
    parser.add_argument('-lr',"--learning-rate",type=float,default=1e-3,help="learning-rate")
    parser.add_argument("--data-folder",type=str,default='./data',help="data_folder")
    parser.add_argument("--save-folder",type=str,default='./logs',help="save folder")

    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    args = parser.parse_args()

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        CPCGridMaker((args.crop_size,args.crop_size))
        ])

    minist_train = datasets.MNIST(args.data_folder, train=True, download=True,
                       transform=transform, shuffle = True)
    minist_test = datasets.MNIST(args.data_folder, train=False,
                       transform=transform)
    
    
    args.latent_size = 512
    train_loader = torch.utils.data.DataLoader(minist_train,batch_size=args.batch_size)
    model = Conv4(K= args.K,latent_size= args.latent_size).to(device)
    model = model.double()
    optimizer = optim.Adam(model.parameters(),weight_decay=args.weight_decay)

    if args.resume_model != None:
        ckpt = torch.load(args.resume_model)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
    for e in range(args.epochs):
        train(e, model, train_loader, device, args, optimizer, logging)
        if args.save_model:
            if (e+1) % 10 == 0:
                torch.save({
                    "model":model.state_dict(),
                    "opt":optimizer.state_dict(),
                    "K" : args.K,
                    "epoch" : e,
                    "latent_size": args.latent_size,
                    "crop_size":args.crop_size,
                    "args" : args
                    },"mnist_epoch{}_ns{}_k{}.pt".format(e,args.num_neg_samples,args.K))
                   

if __name__ == '__main__':
    main()