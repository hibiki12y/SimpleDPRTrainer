import torch

def mean_pooling(feature,mask):
    return torch.sum(feature * mask[:,:,None], dim=1) / torch.sum(mask[:,:,None],dim=1)