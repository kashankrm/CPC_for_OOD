import torch

def contrastive_loss(positive, negatives, W, context, temp = 0.5,norm=True,indivisual_loss =False):
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
    if indivisual_loss:
        return torch.tensor((-1/(negatives.shape[1]+1))) * (torch.log(numerator/(denom)))
    else:
        return torch.tensor((-1/(negatives.shape[1]+1))) * (torch.log(numerator/(denom))).sum()
