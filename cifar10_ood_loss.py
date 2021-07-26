from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from minst_model import mnist_Conv4
from model_resnet18_gru import Conv4, Conv4Mini
import logging
from utils import CPCGridMaker
import os




def contrastive_loss(positive, W, context, temp = 0.5,norm=False):
    if len(positive.shape)!=3:
        positive = positive.unsqueeze(dim=1)
    
    c_w = W(context)
    numerator = torch.bmm(positive, c_w.unsqueeze(dim=2)).squeeze().double()
    if norm:
        numerator = numerator/((torch.norm(positive,dim=(1,2)) * torch.norm(c_w,dim=1))+torch.tensor(1e-6))
    numerator = torch.exp(numerator)
    return (torch.log(numerator))
    return numerator

def contrastive_loss_neg(positive, negatives, W, context, temp = 0.5,norm=False):
    if len(positive.shape)!=3:
        positive = positive.unsqueeze(dim=1)
    
    c_w = W(context)
    numerator = torch.bmm(positive, c_w.unsqueeze(dim=2)).squeeze().double()
    denom = torch.bmm(negatives, c_w.unsqueeze(dim=2)).double().squeeze()
    if norm:
        numerator = numerator/((torch.norm(positive,dim=(1,2)) * torch.norm(c_w,dim=1))+torch.tensor(1e-6))
        denom = denom/((torch.norm(negatives,dim=2)* torch.norm(c_w,dim=1).unsqueeze(1))+torch.tensor(1e-6))

    numerator = torch.exp(numerator)
    denom = torch.sum(torch.exp(denom),dim=1)
    
    return torch.tensor((-1/(negatives.shape[1]+1))) * (torch.log(numerator/(denom+numerator)))

def get_ood_loss(loader, model, args):
    concat_loss = None
    with torch.no_grad():
        for batch_idx,(data,true_pred) in enumerate(loader):
            cur_batch = data.shape[0]
            data = data.to(args.device).double()
            loss = torch.zeros(true_pred.shape[0]).to(args.device)
            grid_shape,img_shape = data.shape[:3],data.shape[3:]
            grid_shape_x = grid_shape[-1]
            output = model(data.view(-1,*img_shape))
            output = output.view(*grid_shape,-1)
            for t in range(grid_shape_x-args.K):
                for c in range(grid_shape_x):        
                    enc_grid = output[:,:t+1,:,:].view(cur_batch,-1,model.latent_size)
                    enc_grid = enc_grid[:,:-(grid_shape_x-c-1) if (c <grid_shape_x-1) else grid_shape_x,:]
                    ar_out,_ = model.auto_regressive(enc_grid)
                    ar_out = ar_out[:,-1,:]
                    targets = output[:,t+1:t+args.K+1,c,:]
                    for k in range(args.K):
                        pos_sample = targets[:,k,:] 
                        loss += contrastive_loss(pos_sample,model.W[k],ar_out,norm=False)
            concat_loss = np.concatenate((concat_loss, loss.cpu().numpy())) if np.any(concat_loss != None) else loss.cpu().numpy()
    return concat_loss

def get_histograms(id_loader, ood_loader, model, in_name, out_name, args):
    out_hist = get_ood_loss(ood_loader, model, args)
    in_hist = get_ood_loss(id_loader, model, args)
    print("In distribution data loss mean = ", np.mean(in_hist))
    print("In distrubtion data loss var = ", np.var(in_hist))
    print("In distrubtion data loss std = ", np.std(in_hist))
    print("OOD data loss mean = ", np.mean(out_hist))
    print("OOD data loss var = ", np.var(out_hist))
    print("OOD data loss std = ", np.std(out_hist))
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    df = pd.concat(axis=0, ignore_index=True, objs=[
        pd.DataFrame.from_dict({'value': in_hist, 'name': in_name}),
        pd.DataFrame.from_dict({'value': out_hist, 'name': out_name})
    ])
    g = sns.histplot(data=df, x='value', hue='name', multiple='dodge')
    g.legend_.set_title(None)
    plt.axvline(np.mean(in_hist),color='blue', linestyle='--')
    plt.axvline(np.mean(out_hist),color='red', linestyle='--')
    plt.savefig(os.path.join(args.graph_folder, args.name))



def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-k','--K', type=int, default=3 ,
                        help='how many steps to predict')
    parser.add_argument('-rm',"--resume-model",type=str,help="path to ckpt to resume model")
    parser.add_argument('-cs',"--crop-size",type=int,default=8,help="crop size")
    parser.add_argument('-bs','--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument("--data-folder",type=str,default='./data',help="data_folder")
    parser.add_argument("--logs", type=str, default="./logs", help="folder to save histograms")
    parser.add_argument("--name", type=str, default="histogram", help="name to save histogram as")
    args = parser.parse_args()
    args.data_folder = "/misc/student/mirfan/CPC_for_OOD/data"
    args.logs = "/misc/student/mirfan/ood_loss/logs"
    args.name = "EmnistonMnist_mnist_epoch89_ns30_k3"
    args.latent_size = 512
    args.K = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.graph_folder = os.path.join(args.logs, "graph")
    if not os.path.isdir(args.graph_folder):
        os.makedirs(args.graph_folder)

    args.resume_model = "/misc/student/mirfan/ood_loss/models/mnist_epoch89_ns30_k3.pt"
    model = mnist_Conv4(latent_size=args.latent_size, K = args.K).to(device)
    # model = Conv4(img_channels=3,K=args.K,hidden_size=100,layers=1,latent_size=512).to(device)
    model = model.double()

    ckpt = torch.load(args.resume_model)
    new_state_dict = {}
    for k, v in ckpt["model"].items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.feature = torch.nn.DataParallel(model.feature).to(device)
    model.auto_regressive = torch.nn.DataParallel(model.auto_regressive).to(device)
    get_histograms(get_mnist_loader(args),get_emnist_loader(args),  model, "mnist", "emnist", args)
    

if __name__ == '__main__':
    main()