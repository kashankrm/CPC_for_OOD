from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from loss import contrastive_loss
from models_cifar10 import Conv4
import logging
from CPCGridMaker import CPCGridMaker
import os
from yellowbrick.text import TSNEVisualizer
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-bs','--batch-size', type=int, default=128, metavar='N',
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
    args.model_path = os.path.join(args.save_folder , 'models')
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435, 0.2616)),
        CPCGridMaker((args.crop_size, args.crop_size))
    ])
    args.data_folder = "/misc/student/mirfan/CPC_for_OOD/data"
    args.model = 'resnet18'
    args.resume_model = "/misc/student/mirfan/CPC_for_OOD/logs/models/cifar10_epoch52_bs512_ns40.pt"
    cifar_train = datasets.CIFAR10(args.data_folder, train=False, download=True,
                       transform=transform)
    test_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.batch_size, num_workers = 20)
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

    model.feature = torch.nn.DataParallel(model.feature).to(device)
    model.auto_regressive = torch.nn.DataParallel(model.auto_regressive).to(device)
    latent_size = model.latent_size
    out_data ={}
    for batch_idx,(x,target) in enumerate(test_loader):
        print(batch_idx)
        batch_grid, img_shape = x.shape[:3],x.shape[3:]
        x = x.to(device).double()
        res = model.feature(x.view(-1,*img_shape))
        res_flat = res.view(batch_grid[0],-1)
        output = res.view(*batch_grid,-1)
        output = model.auto_regressive(output[:,:,:,:].view(output.shape[0],-1,512))[0].view(output.shape[0],-1)
        concatenated_features = torch.cat((output,res_flat), 1)
        for i,t in enumerate(target):
            k = t.item()
            out_data[k] = out_data.get(k,[])
            # if len(out_data[k]) > 100 :
            #     continue
            out_data[k].append(concatenated_features[i].detach().cpu().numpy()) 
    X = []
    y = []
    for key in out_data:
        X = X + out_data[key]
        y = y + [key]*len(out_data[key])
        out_data[key] = np.stack(out_data[key],axis=0)
    X = np.stack(X,axis=0)
    y = np.stack(y,axis=0)
    y = [str(i) for i in y]
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    pca_components = PCA(n_components=2).fit_transform(X)
    tsne_components_2 = TSNE(n_components=2).fit_transform(X)
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.scatterplot(x=tsne_components_2[:,0] , y=tsne_components_2[:,1] , hue=y)
    plt.savefig("/misc/student/mirfan/epoch52_cifar10_all_tsne.jpeg")
    plt.clf()
    sns.scatterplot(x=pca_components[:,0] , y=pca_components[:,1] , hue=y)
    plt.savefig("/misc/student/mirfan/epoch52_cifar10_all_pca.jpeg")
    plt.clf()
    del X,y, concatenated_features, out_data


    cifar_train = datasets.CIFAR100(args.data_folder, train=False, download=True,
                       transform=transform)
    test_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.batch_size, num_workers = 20)
    # out_data ={}
    # for batch_idx,(x,target) in enumerate(test_loader):
    #     print(batch_idx)
    #     batch_grid, img_shape = x.shape[:3],x.shape[3:]
    #     x = x.to(device).double()
    #     res = model.feature(x.view(-1,*img_shape))
    #     res_flat = res.view(batch_grid[0],-1)
    #     output = res.view(*batch_grid,-1)
    #     output = model.auto_regressive(output[:,:,:,:].view(output.shape[0],-1,512))[0].view(output.shape[0],-1)
    #     concatenated_features = torch.cat((output,res_flat), 1)
    #     for i,t in enumerate(target):
    #         k = t.item()
    #         out_data[k] = out_data.get(k,[])
    #         out_data[k].append(concatenated_features[i].detach().cpu().numpy()) 
    # X = []
    # y = []
    # for key in out_data:
    #     X = X + out_data[key]
    #     y = y + [key]*len(out_data[key])
    #     out_data[key] = np.stack(out_data[key],axis=0)
    # X = np.stack(X,axis=0)
    # y = np.stack(y,axis=0)
    # y = [str(i) for i in y]
    # from sklearn.manifold import TSNE
    # from sklearn.decomposition import PCA
    # pca_components = PCA(n_components=2).fit_transform(X)
    # tsne_components_2 = TSNE(n_components=2).fit_transform(X)
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.scatterplot(x=tsne_components_2[:,0] , y=tsne_components_2[:,1] , hue=y)
    # plt.savefig("/misc/student/mirfan/epoch52_cifar100_all_tsne.jpeg")
    # plt.clf()
    # sns.scatterplot(x=pca_components[:,0] , y=pca_components[:,1] , hue=y)
    # plt.savefig("/misc/student/mirfan/epoch52_cifar100_all_pca.jpeg")
    # plt.clf()
    # del X,y, concatenated_features, out_data





    cifar_train = datasets.CIFAR10(args.data_folder, train=False, download=True,
                       transform=transform)
    test_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.batch_size, num_workers = 20)
    out_data ={}
    for batch_idx,(x,target) in enumerate(test_loader):
        print(batch_idx)
        batch_grid, img_shape = x.shape[:3],x.shape[3:]
        x = x.to(device).double()
        res = model.feature(x.view(-1,*img_shape))
        res_flat = res.view(batch_grid[0],-1)
        for i,t in enumerate(target):
            k = t.item()
            out_data[0] = out_data.get(0,[])
            # if len(out_data[k]) > 10 :
            #     continue
            out_data[0].append(res_flat[i].detach().cpu().numpy()) 

    cifar_train = datasets.CIFAR100(args.data_folder, train=False, download=True,
                       transform=transform)
    test_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.batch_size, num_workers = 20)
    for batch_idx,(x,target) in enumerate(test_loader):
        print(batch_idx)
        batch_grid, img_shape = x.shape[:3],x.shape[3:]
        x = x.to(device).double()
        res = model.feature(x.view(-1,*img_shape))
        res_flat = res.view(batch_grid[0],-1)
        for i,t in enumerate(target):
            k = t.item()+10
            out_data[1] = out_data.get(1,[])
            # if len(out_data[k]) > 10 :
            #     continue
            out_data[1].append(res_flat[i].detach().cpu().numpy()) 
    X = []
    y = []
    for key in out_data:
        X = X + out_data[key]
        y = y + [key]*len(out_data[key])
        out_data[key] = np.stack(out_data[key],axis=0)
    X = np.stack(X,axis=0)
    y = np.stack(y,axis=0)
    y = [str(i) for i in y]
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    pca_components = PCA(n_components=2).fit_transform(X)
    tsne_components_2 = TSNE(n_components=2).fit_transform(X)
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.scatterplot(x=tsne_components_2[:,0] , y=tsne_components_2[:,1] , hue=y)
    plt.savefig("/misc/student/mirfan/trying_tsne.jpeg")
    plt.clf()
    sns.scatterplot(x=pca_components[:,0] , y=pca_components[:,1] , hue=y)
    plt.savefig("/misc/student/mirfan/trying_pca.jpeg")
    plt.clf()


if __name__ == '__main__':
    main()