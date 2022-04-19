import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, feature1, feature2, label):
        dists = F.pairwise_distance(feature1, feature2, p=2.0)
        label = (label==0).to(dtype=torch.float32)
        
        #print('dist : ',dists)
        #print('label : ',label)
        loss = torch.mean(label*(dists**2)+(1-label)*(torch.clamp(self.margin-dists, min=0.0)**2))
        return loss
