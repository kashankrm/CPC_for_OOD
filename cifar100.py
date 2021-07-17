from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from loss import contrastive_loss
from models_cifar100 import Conv4
import logging
from CPCGridMaker import CPCGridMaker
import os

def main():

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-bs','--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-m','--model', type=str, default='resnet18', 
                        help='Model name (default: resnet18)')
    parser.add_argument('-li','--logging-interval', type=int, default=10,
                        help='how often to print loss, every nth')
    parser.add_argument('-sm','--save-model', type=bool, default=True ,
                        help='should model be saved')
    parser.add_argument('-e','--epochs', type=int, default=100 ,
                        help='how many epochs to run')
    parser.add_argument('-ns','--num_neg_samples', type=int, default=30 ,
                        help='how many negative samples to use')
    parser.add_argument('-wd',"--weight-decay",type=float,default=1e-5,
                        help=" weight decay for adam")
    parser.add_argument('-k','--K', type=int, default=3 ,
                        help='how many steps to predict')
    parser.add_argument('-rm',"--resume-model",type=str,help="path to ckpt to resume model")
    parser.add_argument('-cs',"--crop-size",type=int,default=8,help="crop size")
    parser.add_argument('-lr',"--learning-rate",type=float,default=0.001,help="learning-rate")
    parser.add_argument("--data-folder",type=str,default='./data',help="data_folder")
    parser.add_argument("--save-folder",type=str,default='./logs',help="save folder")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # args.data_folder = "/misc/student/mirfan/CPC_for_OOD/data"
    # args.model = 'resnet50'
    args.logfile_name = os.path.join(args.save_folder , 'cifar100_ns{}_k{}.log'.format(args.num_neg_samples,args.K))
    args.model_path = os.path.join(args.save_folder , 'models')
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        CPCGridMaker((args.crop_size, args.crop_size))
    ])

    cifar_train = datasets.CIFAR100(args.data_folder, train=True, download=True,
                       transform=transform)

    K=args.K
    num_neg_sample = args.num_neg_samples
    train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.batch_size, num_workers = 20)
    model = Conv4(name = args.model, K = args.K).to(device)
    optimizer = optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr = args.learning_rate)
    model = model.double()
    if args.resume_model != None:
        ckpt = torch.load(args.resume_model)
        new_state_dict = {}
        for k, v in ckpt["model"].items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(ckpt["opt"])
    model.feature = torch.nn.DataParallel(model.feature).to(device)
    model.auto_regressive = torch.nn.DataParallel(model.auto_regressive).to(device)
    latent_size = model.latent_size
    
    logging.basicConfig(filename=args.logfile_name, level=logging.DEBUG)

    for e in range(0,args.epochs):
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
            if e%10:
                torch.save({
                    "model":model.state_dict(),
                    "opt":optimizer.state_dict(),
                    "K" : args.K,
                    "epoch" : e,
                    "latent_size": latent_size,
                    "crop_size" : args.crop_size,
                    "model_name" : args.model,
                    "batch_size" : args.batch_size,
                    "num_neg_samples" : args.num_neg_samples

                },os.path.join(args.model_path, "cifar100_epoch{}_bs{}_ns{}.pt".format(e, args.batch_size, args.num_neg_samples)))
    
if __name__ == '__main__':
    main()