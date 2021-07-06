from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from loss import contrastive_loss
from pytorch_lightning import loggers as pl_loggers
import sys
sys.path.append("/project/dl2021s/mirfan/CPC_for_OOD/")
from models import Conv4, LinClassifier
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging
logging.basicConfig(filename='cifar10classifier.log', level=logging.DEBUG)
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
    parser.add_argument('-bs','--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-sm','--save-model', type=bool, default=True ,
                        help='should model be saved')
    parser.add_argument('-e','--epochs', type=int, default=500 ,
                        help='how many epochs to run')
    parser.add_argument("-pt","--pretrain",type =str,help="path to the pretrain network")
    parser.add_argument('-li','--logging-interval', type=int, default=10 ,
                        help='how often to print loss, every nth')            
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    args = parser.parse_args()
    email_sara_mila_lo = True
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
        CPCGridMaker((8,8))
        ])
    cifar_train = datasets.CIFAR10('./data', train=True, download=True,
                       transform=transform)
    cifar_test = datasets.CIFAR10('./data', train=False,
                       transform=transform)t

    train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.batch_size,num_workers=4 )
    test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=args.batch_size,num_workers=4 )
    model = LinClassifier("/project/dl2021s/mirfan/CPC_for_OOD/cifarpixelAsmaa_epoch12.pt")
    from pytorch_lightning.callbacks import ModelCheckpoint
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.0,
        patience=50,
        verbose=True,
        mode='max'
        )
    tb_logger = pl_loggers.TensorBoardLogger('logs/',name="cifarpixelAsmaa10runs")
    trainer = pl.Trainer(gpus=1,max_epochs= args.epochs,min_epochs=1,logger=tb_logger, callbacks=[early_stop_callback])
    trainer.fit(model, train_loader, test_loader)

if __name__ == '__main__':
    main()