#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:17:18 2021

@author: sf00511
"""
import torch
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from torch.nn import Parameter
import torch.nn as nn
import math
from Database import return_augmented_test,return_augmented_train

    
         
def train_epoch(model, epoch, optimizer, optimizer_center, criterion_center, T, data_loader, GT_classes, device):
    transformer_bank=[]
    tr_idx=0
    for i in range(0,2):
        for w in range(0,4):
            transformer_bank.append([i,w,tr_idx])
            tr_idx=tr_idx+1
    
    data_len=len(data_loader.dataset)
    idx_bank=np.tile(np.arange(GT_classes), data_len)
    idx_tr=np.arange(GT_classes)
    for idx_data_len in range(data_len):
        np.random.shuffle(idx_tr)
        ct=0
        for qq in range(GT_classes):
            idx_bank[(data_len*ct)+idx_data_len]=idx_tr[qq]
            ct+=1
            
    total_samples = len(data_loader.dataset)
    model.train()
    total_loss=0.0
    total_term1=0.0
    total_term2=0.0
    
    
    
    if epoch<5:
        w_center=0.02
        w_cross=0.98
    elif epoch>=5 and epoch<15:
        w_center=0.05
        w_cross=0.95
    elif epoch>=15 and epoch<30:
        w_center=0.08
        w_cross=0.92
    else:
        w_center=0.1
        w_cross=0.9
    
    print("*********************")
    for times in range(GT_classes):
        reach_id=0

        for data, target in data_loader:
            sample_len=len(data)
            orig_data=data
            orig_target=target
            
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            start_idx=(times*data_len)+(sample_len*reach_id)
            end_idx=(times*data_len)+(sample_len*reach_id)+sample_len-1
            another_cpu_data,another_cpu_target=return_augmented_train(orig_data,orig_target,transformer_bank,idx_bank[start_idx:end_idx+1])
            reach_id+=1
                       
            with torch.set_grad_enabled(True):
                features, logits = model(another_cpu_data)
                term1 = F.cross_entropy(logits/T, another_cpu_target)
                term2 = criterion_center(features.cuda(), another_cpu_target.to(device),epoch)

                
                        
                loss=(term1*w_cross)+(term2*w_center)
                loss.backward()
                
                            
                optimizer.step()
                optimizer_center.step()
                
            total_loss+=loss*data.size(0)
            total_term1+=term1*data.size(0)
            total_term2+=term2*data.size(0)

    
    print('Total Loss:' , total_loss.item()/(total_samples*GT_classes))
    print('Sotmax Cross Entropy Loss:' , total_term1.item()/(total_samples*GT_classes))
    print('AMTCL Loss:' , total_term2.item()/(total_samples*GT_classes))
    
    return total_loss.item()/(total_samples*GT_classes)
                
