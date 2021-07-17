from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from pytorch_lightning import loggers as pl_loggers
import sys
from models_cifar100 import LinClassifier
from CPCGridMaker import CPCGridMaker
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging


def main():

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-bs','--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('-sm','--save-model', type=bool, default=True ,
                        help='should model be saved')
    parser.add_argument('-e','--epochs', type=int, default=200 ,
                        help='how many epochs to run')
    parser.add_argument("-pt","--pretrain",type =str,help="path to the pretrain network")
    parser.add_argument('-li','--logging-interval', type=int, default=10 ,
                        help='how often to print loss, every nth')     
    parser.add_argument('-rm','--resume-model', type=str,
                        help='Resume model')       
    args = parser.parse_args()
    ckpt = torch.load(args.pretrain)

    if "crop_size" in ckpt: 
        args.crop_size = ckpt["crop_size"]
    else:
        args.crop_size = 8
    logging.basicConfig(filename='cifar100classifier_bs{}_k{}.log'.format(args.batch_size,ckpt["K"]), level=logging.DEBUG)    
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    CPCGridMaker((args.crop_size,args.crop_size))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        CPCGridMaker((args.crop_size,args.crop_size))
    ])
    cifar_train = datasets.CIFAR100('./data', train=True, download=True,
                       transform=transform_train)
    cifar_test = datasets.CIFAR100('./data', train=False,
                       transform=transform_test)
    train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.batch_size,num_workers=4 )
    test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=args.batch_size,num_workers=4 )
    model = LinClassifier(args.pretrain, 100)
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.0,
        patience=50,
        verbose=True,
        mode='max'
        )
    tb_logger = pl_loggers.TensorBoardLogger('./logs/',name="cifar100")
    trainer = pl.Trainer(gpus=1,max_epochs= args.epochs,min_epochs=1,logger=tb_logger, callbacks=[early_stop_callback])
    trainer.fit(model, train_loader, test_loader)

if __name__ == '__main__':
    main()