import torch
import numpy as np
from loss import contrastive_loss

class CPCGridMaker:
    def __init__(self,grid_shape):
        self.grid_shape = grid_shape
        
    def __call__(self,sample):
        
        sample = torch.squeeze(sample)
        grid_x,grid_y = self.grid_shape
        d, h, w = sample.shape if len(sample.shape)>2 else (0,*sample.shape)
        out_shape_y,out_shape_x = h//(grid_y//2) -1, w//(grid_x//2) -1
        
        if d == 0:
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


def train(epoch, model, train_loader, device, args, optimizer, logging):
    model.train()
    total_loss = 0.0
    num_samples = 0
    for batch_idx,(data,_) in enumerate(train_loader):
        cur_batch = data.shape[0]
        data = data.to(device).double()
        loss = torch.tensor(0.0).to(device)
        num_samples += data.shape[0]
        grid_shape,img_shape = data.shape[:3],data.shape[3:]
        grid_size = grid_shape[-1]
        output = model(data.view(-1,*img_shape))
        output = output.view(*grid_shape,-1)
        for t in range(grid_size-args.K):
            for c in range(grid_size):        
                enc_grid = output[:,:t+1,:,:].view(cur_batch,-1, model.latent_size)
                enc_grid = enc_grid[:,:-(grid_size-c-1) if (c <grid_size-1) else grid_size,:]
                ar_out,_ = model.auto_regressive(enc_grid)
                ar_out = ar_out[:,-1,:]
                targets = output[:,t+1:t+args.K+1,c,:]
                for k in range(args.K):
                    pos_sample = targets[:,k,:] 
                    neg_sample_idx = np.random.choice(grid_size**2,args.num_neg_samples,replace=True)
                    neg_samples = output.view(cur_batch,-1, model.latent_size)[:,neg_sample_idx,:]
                    loss += contrastive_loss(pos_sample,neg_samples,model.W[k],ar_out,norm=True)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % args.logging_interval == 0:
            print("average Loss is {:.4f}, batch_idx is {}/{}".format(loss.item()/data.shape[0],batch_idx,len(train_loader)))
            logging.debug("average Loss is {:.4f}, batch_idx is {}/{}".format(loss.item()/data.shape[0],batch_idx,len(train_loader)))
    print("Loss is {}, epoch is {}".format(total_loss/num_samples, epoch))
    logging.debug("Loss is {}, epoch is {}".format(total_loss/num_samples, epoch))


def get_OmniGlot_loader(args):
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
        transforms.Normalize(mean=[0.92206], std=[0.08426]),
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

def get_cifar10_loader(args, train = False):
    if train:
        transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        CPCGridMaker((args.crop_size, args.crop_size))
    ])
    else:
        transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435, 0.2616)),
        CPCGridMaker((args.crop_size, args.crop_size))
    ])
    
    cifar_test = datasets.CIFAR10(args.data_folder, train=train, download=True,
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