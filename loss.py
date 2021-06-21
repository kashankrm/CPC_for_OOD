import torch

def contrastive_loss(positive, negatives, W, context, temp = 0.5,norm=True):
    if len(positive.shape)!=3:
        positive = positive.unsqueeze(dim=1)
    
    c_w = W(context)
    # >>> positive.shape
    # torch.Size([10, 1024])
    # >>> negative.shape
    # torch.Size([10, 5, 1024])
    # >>> c_w.shape
    # torch.Size([10, 1024])

    numerator = torch.bmm(positive, c_w.unsqueeze(dim=2)).squeeze().double()
    denom = torch.bmm(negatives, c_w.unsqueeze(dim=2)).double().squeeze()
    if norm:
        numerator = numerator/((torch.norm(positive,dim=(1,2)) * torch.norm(c_w,dim=1))+torch.tensor(1e-6))
        denom = denom/((torch.norm(negatives,dim=2)* torch.norm(c_w,dim=1).unsqueeze(1))+torch.tensor(1e-6))

    numerator = torch.exp(numerator)
    denom = torch.sum(torch.exp(denom),dim=1)
    
    return torch.tensor((-1/(negatives.shape[1]+1))) * (torch.log(numerator/(denom+numerator))).sum()
