from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from loss import contrastive_loss
from models import Conv4
# from feature_bank import FeatureBank

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
    parser.add_argument('-bs','--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-li','--logging-interval', type=int, default=10 ,
                        help='how often to print loss, every nth')
    parser.add_argument('-sm','--save-model', type=bool, default=True ,
                        help='should model be saved')
    parser.add_argument('-e','--epochs', type=int, default=200 ,
                        help='how many epochs to run')
    parser.add_argument('-ns','--num_neg_samples', type=int, default=10 ,
                        help='how many negative samples to use')
    parser.add_argument('-wd',"--weight-decay",type=float,default=1e-5,
                        help=" weight decay for adam")
                           
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # device = torch.device("cpu")
    

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        CPCGridMaker((8,8))
        ])
    cifar_train = datasets.CIFAR10('./data', train=True, download=True,
                       transform=transform)
    cifar_test = datasets.CIFAR10('./data', train=False,
                       transform=transform)
    grid_shape_x = 7
    K=2
    num_neg_sample = args.num_neg_samples
    latent_size = 512
    email_sara_mila_lo = True
    
    train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(cifar_test)

    model = Conv4(img_channels=3,K=K,latent_size=latent_size).to(device)
    model = model.double()
    optimizer = optim.Adam(model.parameters(),weight_decay=args.weight_decay)
    
    for e in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_samples = 0
        # feature_bank = FeatureBank(max_len=10)
        for batch_idx,(data,_) in enumerate(train_loader):
            cur_batch = data.shape[0]
            data = data.to(device).double()
            loss = torch.tensor(0.0).to(device)
            num_samples += data.shape[0]
            grid_shape,img_shape = data.shape[:3],data.shape[3:]
            output = model(data.view(-1,*img_shape))
            output = output.view(*grid_shape,-1)
            # feature_bank.append(output.detach().cpu().numpy(),batch_idx)
            for c in range(grid_shape_x):
                cl = output[:,:,c,:]
                for t in range(grid_shape_x-K):
                    inp = cl[:,:t+1,:]
                    ar_out,_ = model.auto_regressive(inp)
                    ar_out = ar_out[:,-1,:]
                    targets = cl[:,t+1:t+K+1,:]
                    for k in range(K):
                        pos_sample = targets[:,k,:] 
                        neg_idx = [i for i in range(grid_shape_x) if i!=c]
                        # feats = feature_bank.get_samples(num_neg_sample,batch_idx)
                        # ncl_data = np.stack(feats,axis=1)
                        if email_sara_mila_lo:
                            ncl = output[:,:,neg_idx,:]
                            total_neg_sample = ncl.shape[1]*ncl.shape[2]
                        else:
                            ncl = output[:,k+t,neg_idx,:]
                            total_neg_sample = ncl.shape[1]
                        neg_samples_arr = ncl.reshape((-1,total_neg_sample,latent_size))

                        neg_sample_idx = np.random.choice(total_neg_sample,num_neg_sample,replace=True)
                        neg_samples = neg_samples_arr[:cur_batch,neg_sample_idx,:]
                        loss += contrastive_loss(pos_sample,neg_samples,model.W[k],ar_out,norm=True)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % args.logging_interval ==0:
                print("average Loss is {:.4f}, batch_idx is {}/{}".format(loss.item()/data.shape[0],batch_idx,len(train_loader)))
        print("Loss is {}, epoch is {}".format(total_loss/num_samples,e))
        if args.save_model:
            torch.save({
                "model":model.state_dict(),
                "opt":optimizer.state_dict()
            },"cifar_epoch{}.pt".format(e))
                        
if __name__ == '__main__':
    main()