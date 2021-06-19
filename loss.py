import torch

def contrastive_loss(positive, negatives, W, context, temp = 0.5):
    if len(positive.shape)!=3:
        positive = positive.unsqueeze(dim=1)
    loss = torch.zeros(0)
    c_w = W(context)
    numerator = torch.exp(torch.bmm(positive, c_w.unsqueeze(dim=2)).squeeze(1).double())
    denom = torch.sum(torch.exp(torch.bmm(negatives, c_w.unsqueeze(dim=2)).double()),dim=1)
    return torch.tensor((-1/(negatives.shape[1]+1))) * (torch.log(numerator/denom))
