from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from minst_model import mnist_Conv4
from model_resnet18_gru import Conv4, Conv4Mini, fConv4
import logging
from utils import CPCGridMaker
import os

def get_omniglot_loader(args):
    stat_transform = transforms.Compose([
        transforms.ToTensor()
    ])        
    cifar_test = datasets.Omniglot(args.data_folder, background = False, download=True,
                       transform=stat_transform)
    stat_loader = torch.utils.data.DataLoader(cifar_test,batch_size=args.batch_size, num_workers = 4)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, labels in stat_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples    
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        lambda x:transforms.functional.invert(x),
        transforms.Normalize(mean=[mean], std=[std]),
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(28),
        CPCGridMaker((args.crop_size, args.crop_size))
    ])
    cifar_test = datasets.Omniglot(args.data_folder, background = False, download=True,
                       transform=transform)
    return torch.utils.data.DataLoader(cifar_test,batch_size=args.batch_size, num_workers = 4)


def get_emnist_loader(args):
    stat_transform = transforms.Compose([
        transforms.ToTensor()
    ])        
    cifar_test = datasets.EMNIST(args.data_folder, split = "letters", train=False, download=True,
                       transform=stat_transform)
    stat_loader = torch.utils.data.DataLoader(cifar_test,batch_size=args.batch_size, num_workers = 4)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, labels in stat_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples    
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
        CPCGridMaker((args.crop_size, args.crop_size))
    ])
    cifar_test = datasets.EMNIST(args.data_folder, split = "letters", train=False, download=True,
                       transform=transform)


    
    return torch.utils.data.DataLoader(cifar_test,batch_size=args.batch_size, num_workers = 4)

def get_cifar10_loader(args):
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.Grayscale(num_output_channels=1),
        CPCGridMaker((args.crop_size, args.crop_size))
    ])
    cifar_test = datasets.CIFAR10(args.data_folder, train= False, download=True,
                       transform=transform)
    return torch.utils.data.DataLoader(cifar_test,batch_size=args.batch_size, num_workers = 4)

def get_mnist_loader(args, train = False):
    transform=transforms.Compose([
    # transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081)),
    CPCGridMaker((args.crop_size,args.crop_size))
    ])
    minist_test = datasets.MNIST(args.data_folder, train=train, download = True,
                       transform=transform)
    return torch.utils.data.DataLoader(minist_test,batch_size=args.batch_size, num_workers = 4)

def get_cifar100_loader(args, train = False):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        CPCGridMaker((args.crop_size,args.crop_size))
    ])
    cifar_test = datasets.CIFAR100(args.data_folder, train=train,
                       transform=transform_test, download = True)
    return torch.utils.data.DataLoader(cifar_test,batch_size=args.batch_size,num_workers=4 )


def contrastive_loss(positive, W, context, temp = 0.5,norm=False):
    if len(positive.shape)!=3:
        positive = positive.unsqueeze(dim=1)
    
    c_w = W(context)
    numerator = torch.bmm(positive, c_w.unsqueeze(dim=2)).squeeze().double()
    if norm:
        numerator = numerator/((torch.norm(positive,dim=(1,2)) * torch.norm(c_w,dim=1))+torch.tensor(1e-6))
    # return numerator
    numerator = torch.exp(numerator)
    return -1 * (torch.log(numerator))

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
    classes = None
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
            classes = np.concatenate((classes, true_pred.cpu().numpy())) if np.any(classes != None) else true_pred.cpu().numpy()
    return concat_loss, classes

def get_histograms(id_loader, ood_loader, model, in_name, out_name, args):
    out_hist, out_classes = get_ood_loss(ood_loader, model, args)
    in_hist, in_classes = get_ood_loss(id_loader, model, args)
    out_classes = [str(i) for i in out_classes]
    in_classes = [str(i) for i in in_classes]
    s = "\nIn distribution data loss mean = " + str(np.mean(in_hist))
    s += "\nIn distrubtion data loss var = " + str( np.var(in_hist))
    s += "\nIn distrubtion data loss std = " + str(np.std(in_hist))
    s += "\nOOD data loss var = " + str(np.var(out_hist))
    s += "\nIn distrubtion data loss var = " + str(np.var(in_hist))
    s += "\nOOD data loss std = " + str(np.std(out_hist))
    import scipy.stats
    import matplotlib.pyplot as plt
    from sklearn import tree
    import seaborn as sns
    import pandas as pd
    from sklearn.metrics import classification_report, accuracy_score 
    concat_X = np.concatenate([out_hist,in_hist]).reshape(-1,1)
    concat_Y = np.concatenate([[0]*len(out_hist), [1]*len(in_hist)])
    
    threshold = scipy.stats.norm.ppf(.9,loc=np.mean(in_hist), scale=np.std(in_hist))
    y_pred = concat_X < threshold
    acc = accuracy_score(y_pred, concat_Y)
    print(acc)
    print(classification_report(concat_Y, y_pred))
    df = pd.concat(axis=0, ignore_index=True, objs=[
        pd.DataFrame.from_dict({'Loss': in_hist, 'name': in_name}),
        pd.DataFrame.from_dict({'Loss': out_hist, 'name': out_name})
    ])
    g = sns.histplot(data=df, x='Loss', hue='name', multiple='dodge')
    g.legend_.set_title(None)
    plt.axvline(np.mean(in_hist),color='blue', linestyle='--')
    plt.axvline(np.mean(out_hist),color='red', linestyle='--')
    plt.axvline(threshold,color='black', linestyle='solid')
    plt.savefig(os.path.join(args.graph_folder, args.name+"_"+str(int(acc*100))))
    r = open(os.path.join(args.graph_folder, args.name+"_"+str(int(acc*100))), "w")
    r.write(classification_report(concat_Y, y_pred))
    r.write("\n\n")
    r.write("Threshold is = " +  str(threshold))
    r.write(s)
    r.close()
    df.to_csv(os.path.join(args.graph_folder, args.name+"_"+str(int(acc*100))) + ".csv")


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
    args.logs = "/misc/student/mirfan/CPC_for_OOD/logs"
    args.data_folder = "/misc/student/mirfan/CPC_for_OOD/data"
    args.name = "MnistonCifar10_cifar_resnet_epoch89_bs64_ns40_cs8_hs100_g3"
    args.latent_size = 512
    args.K = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.graph_folder = os.path.join(args.logs, "graph")
    if not os.path.isdir(args.graph_folder):
        os.makedirs(args.graph_folder)

    args.resume_model = "/misc/student/mirfan/ood_loss/models/cifar_resnet_epoch89_bs64_ns40_cs8_hs100_g3.pt"
    # model = mnist_Conv4(latent_size=args.latent_size, K = args.K).to(device)
    model = Conv4(img_channels=3,K=args.K,hidden_size=100,layers=3,latent_size=512).to(device)
    # model = fConv4(img_channels=3,K=args.K,hidden_size=100,latent_size=1024).to(device)
    model = model.double()

    ckpt = torch.load(args.resume_model)
    new_state_dict = {}
    for k, v in ckpt["model"].items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.feature = torch.nn.DataParallel(model.feature).to(device)
    model.auto_regressive = torch.nn.DataParallel(model.auto_regressive).to(device)

    # transform=transforms.Compose([
    # # transforms.Lambda(lambda image: image.convert('RGB')),
    # transforms.ToTensor(),
    # transforms.Normalize((0.1307), (0.3081)),
    # CPCGridMaker((args.crop_size,args.crop_size))
    # ])
    # cifar_test = datasets.MNIST(args.data_folder, train= False, download = True,
    #                    transform=transform)
    
    # included_classes = [0,1,2,3,4,5,6]
    # train_subset = [i for i,v in enumerate(cifar_test.targets) if v in included_classes]
    # dataset_train = torch.utils.data.Subset(cifar_test, train_subset)
    # in_dist = torch.utils.data.DataLoader(dataset_train,batch_size=args.batch_size, num_workers = 4)
    # train_subset = [i for i,v in enumerate(cifar_test.targets) if v not in included_classes]
    # dataset_train = torch.utils.data.Subset(cifar_test, train_subset)
    # out_dist = torch.utils.data.DataLoader(dataset_train,batch_size=args.batch_size, num_workers = 4)
    get_histograms(get_cifar10_loader(args),get_mnist_loader(args),  model, "CIFAR-10", "MNIST", args)


if __name__ == '__main__':
    main()