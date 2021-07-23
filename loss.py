import torch
from torch.nn import functional as F

def contrastive_loss(positive, negatives, W, context, temp = 0.5,norm=True,indivisual_loss =False):
    if len(positive.shape)!=3:
        positive = positive.unsqueeze(dim=1)
    
    c_w = W(context)
    # >>> positive.shape
    # torch.Size([10, 1024])
    # >>> negative.shape
    # torch.Size([10, 5, 1024])
    # >>> c_w.shape
    # torch.Size([10, 1024])

    pos_fk = torch.bmm(positive, c_w.unsqueeze(dim=2)).squeeze().double()
    neg_fk = torch.bmm(negatives, c_w.unsqueeze(dim=2)).double().squeeze()
    if norm:
        pos_fk = pos_fk/((torch.norm(positive,dim=(1,2)) * torch.norm(c_w,dim=1))+torch.tensor(1e-6))
        neg_fk = neg_fk/((torch.norm(negatives,dim=2)* torch.norm(c_w,dim=1).unsqueeze(1))+torch.tensor(1e-6))

    pos_fk = pos_fk.unsqueeze(dim=1)
    loss = torch.cat([pos_fk,neg_fk],dim=1)
    loss = F.softmax(loss,dim=1)
    loss = -loss[:,0]
    # pos_fk = torch.exp(pos_fk)
    # neg_fk = torch.exp(neg_fk)
    # denom = torch.sum(neg_fk,dim=1)+pos_fk
    # loss = torch.log(pos_fk/denom)
    # loss += torch.sum(torch.log(neg_fk/denom.unsqueeze(1)),dim=1)       
    if indivisual_loss:
        return loss
    else:
        return loss.sum()/positive.shape[0]
