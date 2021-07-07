from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from loss import contrastive_loss
from models import Conv4Suggested
from tqdm import tqdm
from matplotlib import pyplot as plt

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
def get_dict(score,label):
    in_dict ={}
    for s,l in zip(score,label):
        in_dict[l] = in_dict.get(l,[])
        in_dict[l].append(s)
    return in_dict
def print_dict(_dict):
    r1 = []
    r2 = []
    for cl in _dict:
        r1.append("\t\t{}".format(cl))
        r2.append("\t\t{}".format(round(sum(_dict[cl])/len(_dict[cl]),2)))
    print("".join(r1))
    print("".join(r2))
    
        
def main():

    parser = argparse.ArgumentParser(description='PyTorch ood experiments')
    parser.add_argument('-pt','--pretrain', type=str, default="cpc_models/cifar_epoch115.pt", 
                        help='pretrain network path')
    parser.add_argument('-bs','--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-li','--logging-interval', type=int, default=100 ,
                        help='how often to print loss, every nth')
    parser.add_argument('-sm','--save-model', type=bool, default=True ,
                        help='should model be saved')
    parser.add_argument('-e','--epochs', type=int, default=20 ,
                        help='how many epochs to run')
    parser.add_argument('-ns','--num_neg_samples', type=int, default=5 ,
                        help='how many negative samples to use')
    parser.add_argument('-wd',"--weight-decay",type=float,default=1e-5,
                        help=" weight decay for adam")
                           
    device = torch.device("cuda" if torch.cuda.is_available() and False else "cpu")
    args = parser.parse_args()

    # device = torch.device("cpu")
    

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        CPCGridMaker((8,8))
        ])
    cifar10_train = datasets.CIFAR10('./data', train=True, download=True,
                       transform=transform)
    # cifar10_test = datasets.CIFAR10('./data', train=False,
                    #    transform=transform)
    grid_shape_x = 7
    K=2
    num_neg_sample = args.num_neg_samples
    latent_size = 1024
    included_classes = [0,1,2,3,4,5,6]
    in_train_subset = [i for i,v in enumerate(cifar10_train.targets) if v in included_classes]
    in_train_subset = np.random.choice(in_train_subset,10)
    in_dataset_train = torch.utils.data.Subset(cifar10_train, in_train_subset)

    out_train_subset = [i for i,v in enumerate(cifar10_train.targets) if v not in included_classes]
    out_train_subset = np.random.choice(out_train_subset,10)
    out_dataset_train = torch.utils.data.Subset(cifar10_train, out_train_subset)

    


    in_loader = torch.utils.data.DataLoader(in_dataset_train,batch_size=args.batch_size)
    out_loader = torch.utils.data.DataLoader(out_dataset_train,batch_size=args.batch_size)
    # test_loader = torch.utils.data.DataLoader(cifar10_test)

    model = Conv4Suggested(img_channels=3,K=K,latent_size=latent_size).to(device)
    model = model.double()
    # optimizer = optim.Adam(model.parameters(),weight_decay=args.weight_decay)
    model.load_state_dict(torch.load(args.pretrain)["model"])
    
    total_loss = 0.0
    num_samples = 0
    in_scores = []
    in_label = []

    
    for batch_idx,(data,labels) in enumerate(tqdm(in_loader)):
        cur_batch = data.shape[0]
        data = data.to(device).double()
        num_samples += data.shape[0]
        grid_shape,img_shape = data.shape[:3],data.shape[3:]
        output = model(data.view(-1,*img_shape))
        output = output.view(cur_batch,-1,latent_size)
        output = output/torch.norm(output,dim=[2]).unsqueeze(2)
        dot_prod = torch.bmm(output,output.permute(0,2,1))
        scores = (dot_prod>=0.5).sum(dim=[1,2]) -49
        in_scores_arr =dot_prod.detach().cpu().numpy()
        in_scores = in_scores + scores.cpu().detach().tolist()
        in_label = in_label + labels.tolist()
        in_images = data.cpu().detach().numpy()
        
    out_scores = []
    out_label = []
    for batch_idx,(data,labels) in enumerate(tqdm(out_loader)):
        cur_batch = data.shape[0]
        data = data.to(device).double()
        num_samples += data.shape[0]
        grid_shape,img_shape = data.shape[:3],data.shape[3:]
        output = model(data.view(-1,*img_shape))
        output = output.view(cur_batch,-1,latent_size)
        output = output/torch.norm(output,dim=[2]).unsqueeze(2)
        dot_prod = torch.bmm(output,output.permute(0,2,1))
        out_scores_arr =dot_prod.detach().cpu().numpy()
        scores = (dot_prod>=0.5).sum(dim=[1,2]) - 49
        out_scores = out_scores + scores.cpu().detach().tolist()
        out_label = out_label + labels.tolist()
        out_images = data.cpu().detach().numpy()

    in_dict = get_dict(in_scores,in_label)
    out_dict = get_dict(out_scores,out_label)
    
    for i in range(2):
        for j in range(2):
            for k in range(5):
                plt.subplot(4,5,i*10+j*5 +k +1)
                if i ==0:
                    plt.imshow(in_scores_arr[j*5+k])
                else:
                    plt.imshow(out_scores_arr[j*5+k])
    fig = plt.figure()
    for i in range(2):
        for j in range(2):
            for k in range(5):
                plt.subplot(4,5,i*10+j*5 +k +1)
                if i ==0:
                    img = in_images[j*5+k]
                    img = np.concatenate(img,axis=2)
                    img = np.concatenate(img,axis=2)
                    img = (img - img.min())/(img.max() - img.min())
                    plt.imshow(img.transpose(1,2,0))
                else:
                    img = out_images[j*5+k]
                    img = np.concatenate(img,axis=2)
                    img = np.concatenate(img,axis=2)
                    img = (img - img.min())/(img.max() - img.min())
                    plt.imshow(img.transpose(1,2,0))
    plt.show()

    print("in dict")
    print_dict(in_dict)
    print("out dict")
    print_dict(out_dict)


    in_scores = np.array(in_scores)
    out_scores = np.array(out_scores)
    print("mean in -> {} out -> {}".format(in_scores.mean(),out_scores.mean()))
    print("")
    

    

    
    
    

if __name__ == '__main__':
    main()