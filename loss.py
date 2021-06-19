import torch

def contrastive_loss(positive, negatives, W, context, temp = 0.5):
    if len(positive.shape)!=3:
        positive = positive.unsqueeze(dim=1)
    loss = torch.zeros(0)
    c_w = W(context)
    # >>> positive.shape
    # torch.Size([10, 1024])
    # >>> negative.shape
    # torch.Size([10, 5, 1024])
    # >>> c_w.shape
    # torch.Size([10, 1024])

    numerator = torch.bmm(positive, c_w.unsqueeze(dim=2)).squeeze().double()
    
    numerator = numerator/(torch.norm(positive,dim=(1,2)) * torch.norm(c_w,dim=1))

    numerator = torch.exp(numerator)
    denom = torch.bmm(negatives, c_w.unsqueeze(dim=2)).double().squeeze()/(torch.norm(negatives,dim=2)* torch.norm(c_w,dim=1).unsqueeze(1))

    denom = torch.sum(torch.exp(denom),dim=1)
    return torch.tensor((-1/(negatives.shape[1]+1))) * (torch.log(numerator/denom)).sum()
