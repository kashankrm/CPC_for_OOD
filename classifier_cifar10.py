from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from loss import contrastive_loss
from pytorch_lightning import loggers as pl_loggers
import sys
from models_cifar10 import Conv4, LinClassifier
from CPCGridMaker import CPCGridMaker
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging
from pytorch_lightning.callbacks import ModelCheckpoint

def main():

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-bs','--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('-sm','--save-model', type=bool, default=True ,
                        help='should model be saved')
    parser.add_argument('-e','--epochs', type=int, default=1000 ,
                        help='how many epochs to run')
    parser.add_argument("-pt","--pretrain",type =str,help="path to the pretrain network")
    parser.add_argument('-li','--logging-interval', type=int, default=10 ,
                        help='how often to print loss, every nth')     
    parser.add_argument('-rm','--resume-model', type=str,
                        help='Resume model')       
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    args = parser.parse_args()
    args.pretrain = "/misc/student/mirfan/CPC_for_OOD/logs/models/cifar10_epoch343_bs512_ns40.pt"
    ckpt = torch.load(args.pretrain)
    if "crop_size" in ckpt: 
        crop_size = ckpt["crop_size"]
    else:
        crop_size = 8
    logging.basicConfig(filename='cifar10classifier_bs{}.log'.format(args.batch_size), level=logging.DEBUG)    
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    CPCGridMaker((crop_size,crop_size))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        CPCGridMaker((crop_size,crop_size))
    ])
    ckpt = torch.load("/misc/student/mirfan/CPC_for_OOD/logs/cifar10/version_14/checkpoints/epoch=199-step=39199.ckpt")
    model = LinClassifier(args.pretrain, 10)
    model.load_state_dict(ckpt["state_dict"])
    model.loaded_optimizer_states_dict = ckpt["optimizer_states"]
    cifar_train = datasets.CIFAR10('./data', train=True, download=True,
                       transform=transform_train)
    cifar_test = datasets.CIFAR10('./data', train=False,
                       transform=transform_test)
    train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.batch_size,num_workers=4 )
    test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=args.batch_size,num_workers=4 )

    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.0,
        patience=50,
        verbose=True,
        mode='max'
        )
    tb_logger = pl_loggers.TensorBoardLogger('./logs/',name="cifar10")
    trainer = pl.Trainer(gpus=1,max_epochs= args.epochs,min_epochs=1,logger=tb_logger, callbacks=[early_stop_callback])
    trainer.fit(model, train_loader, test_loader)

if __name__ == '__main__':
    main()