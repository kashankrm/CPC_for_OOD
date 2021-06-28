from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from loss import contrastive_loss
from models import Conv4, Linear_Layer
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
    parser.add_argument('-bs','--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-sm','--save-model', type=bool, default=True ,
                        help='should model be saved')
    parser.add_argument('-e','--epochs', type=int, default=20 ,
                        help='how many epochs to run')
    parser.add_argument("-pt","--pretrain",type =str,help="path to the pretrain network")
    parser.add_argument('-li','--logging-interval', type=int, default=10 ,
                        help='how often to print loss, every nth')            
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    args = parser.parse_args()
    grid_shape_x = 7
    K=2
    latent_size = 512
    email_sara_mila_lo = True
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        CPCGridMaker((8,8))
        ])
    cifar_train = datasets.CIFAR10('./data', train=True, download=True,
                       transform=transform)
    cifar_test = datasets.CIFAR10('./data', train=False,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.batch_size,num_workers=4 )
    test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=args.batch_size,num_workers=4 )
    model = Conv4(img_channels=3,K=K,latent_size=latent_size).to(device)
    ckpt = torch.load("cifar_epoch15.pt")
    model.load_state_dict(ckpt["model"])
    logging.debug("Model loaded")
    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         print("works")
    #         model = torch.nn.DataParallel(model)
    # model = model.double()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    linear = Linear_Layer().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(args.epochs):
        model.eval()
        linear.train()
        total_loss = 0.0
        num_samples = 0
        for batch_idx,(data,labels) in enumerate(train_loader):
            num_samples += data.shape[0]
            data = data.to(device)
            labels = labels.to(device)
            grid_shape,img_shape = data.shape[:3],data.shape[3:]
            feature = model(data.view(-1,*img_shape))
            feature = feature.view(grid_shape[0],-1)
            output = linear(feature)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % args.logging_interval ==0:
                print("average Loss is {:.4f}, batch_idx is {}/{}".format(loss.item()/data.shape[0],batch_idx,len(train_loader)))
                logging.debug("average Loss is {:.4f}, batch_idx is {}/{}".format(loss.item()/data.shape[0],batch_idx,len(train_loader)))
        print("Loss is {}, epoch is {}".format(total_loss/num_samples,e))
        logging.debug("Loss is {}, epoch is {}".format(total_loss/num_samples,e))
        model.eval()
        linear.eval()
        eval_loss = 0.0
        eval_total = 0.0
        for batch_idx,(data,labels) in enumerate(test_loader):
            eval_total += data.shape[0]
            data = data.to(device)
            labels = labels.to(device)
            grid_shape,img_shape = data.shape[:3],data.shape[3:]
            feature = model(data.view(-1,*img_shape))
            feature = feature.view(grid_shape[0],-1)
            output = linear(feature)
            loss = criterion(output, labels)
            eval_loss += loss.item()
        print("Eval Loss is {}, epoch is {}".format(total_loss/eval_total,e))
        logging.debug("Eval Loss is {}, epoch is {}".format(total_loss/eval_total,e))
        if args.save_model:
            torch.save({
                "model":model.state_dict(),
                "opt":optimizer.state_dict()
            },"cifarclassifier_epoch{}.pt".format(e))

if __name__ == '__main__':
    main()