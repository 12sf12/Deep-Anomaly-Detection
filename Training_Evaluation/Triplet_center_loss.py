#DEEP ANOMALY DETECTION by SF12

import torch 
import torch.nn as nn 
import torch.nn.parallel
import torch.nn.functional as F 
from torch.autograd import Variable 
from torch.nn import Parameter 
import numpy as np 
from torch.autograd import Function


################################################################
## Triplet related loss 
################################################################



class AMTCL(nn.Module):
    def __init__(self, device='cuda', num_classes=10, k_val=3, num_dim=384):
        super(AMTCL, self).__init__() 
        self.device=device
        self.num_dim=num_dim
        self.num_classes=num_classes
        #self.margin = margin 
        #self.ranking_loss = nn.MarginRankingLoss(margin=margin) 
        #self.margin = nn.Parameter(torch.tensor(5.0, requires_grad=True))
        self.margin = nn.Parameter(torch.abs(5*torch.randn(1, 1)),requires_grad=True)
        self.k_val=k_val
        self.bias = nn.Parameter(torch.abs(torch.rand(1, 1)),requires_grad=True)
        self.centers = nn.Parameter(torch.randn(num_classes, num_dim))
        self.centers_weights = nn.Parameter(torch.rand(num_classes, num_dim)) # important to check 
    
    def calc_centers(self, which_one):
        dist=np.zeros((self.num_classes,self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                dist[i,j]=torch.sum((2**self.centers_weights[i])*((self.centers[i]-self.centers[j]))**2, 0) 
         
        dist=np.sqrt(dist)    
        dist=np.sort(dist)
        dist=dist[:,which_one]                
        return dist
    
    def forward(self, inputs, targets, epoch_number): 
        centers_dist=self.calc_centers(1)
        batch_size = inputs.size(0)
        #print('batch_size: ', batch_size) 
        #print('Inputs: ', inputs.shape)
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1)) 
        centers_batch = self.centers.cuda().gather(0, targets_expand) # centers batch 
        centers_weights_batch = self.centers_weights.cuda().gather(0, targets_expand)
        
        #print('min: ', self.centers_weights.min().item())

        #print('centers_batch: ', centers_batch.shape)

        # compute pairwise distances between input features and corresponding centers 
        centers_batch_bz = torch.stack([centers_batch]*batch_size)
        centers_weights_batch_bz = torch.stack([centers_weights_batch]*batch_size) 
        inputs_bz = torch.stack([inputs]*batch_size).transpose(0, 1)
        #print('Input_BZ:',inputs_bz.shape)
        #print('Centre bz: ',centers_batch_bz.shape)
        dist = torch.sum((2**centers_weights_batch_bz)*((centers_batch_bz -inputs_bz))**2, 2).squeeze() 
        dist = dist.clamp(min=1e-12).sqrt() # for numerical stability 

        # for each anchor, find the hardest positive and negative 
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        closest_center=[]
        for i in range(batch_size): # for each sample, we compute distance 
            dist_ap.append(dist[i][mask[i]].max()) # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i]==0].min()) # mask[i]==0: negative samples of sample i 
            closest_center.append(centers_dist[targets[i]])
            
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        closest_center=torch.tensor(closest_center)
        # generate a new label y
        # compute ranking hinge loss 
        y = dist_an.data.new() 
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero
        
        
        #sorted_batch,b=torch.sort(torch.abs(dist_an-dist_ap),0)
        #margin=torch.sum(sorted_batch[batch_size-self.k_val:batch_size]).item()+self.bias.item()
        #self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        #loss = self.ranking_loss(dist_an, dist_ap, y)
        
        #return loss
        
        #prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0) # normalize data by batch size 
        #print('Prec is : ', prec.item())
        new_loss=0
        closest_center=closest_center.cuda()
        for i in range(batch_size):
            if dist_an[i]>=closest_center[i]:
                new_loss+=dist_ap[i]
            else:
                new_loss+=dist_ap[i]-dist_an[i]+closest_center[i]
        
        #new_loss=torch.sum((dist_ap-dist_an)+closest_center.cuda())/batch_size
        
        return new_loss/batch_size
