#DEEP ANOMALY DETECTION by SF12

import torch
import time
import torch.optim as optim
import argparse
import os
from Database import GET_SSL_DB
from Execution import train_epoch
from pathlib import Path
import numpy as np

from datetime import datetime as dt
import datetime

import torch.nn as nn
import copy

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision
import torchvision.transforms.functional as F
from wideresnet import WideResNet
from Triplet_center_loss import AMTCL


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"


torch.manual_seed(1371)
def get_args_parser():
    parser = argparse.ArgumentParser('Deep Anomaly Detection', add_help=False)
    parser.add_argument('--depth', default=40, type=int)
    parser.add_argument('--widen_factor', default=4, type=int)
    parser.add_argument('--GT_classes', default=8, type=int)
    parser.add_argument('--dropout', default=3e-1, type=float)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--gamma', default=3e-1, type=float)
    parser.add_argument('--weight_decay_SCE', default=5e-3, type=float)
    parser.add_argument('--weight_decay_AMTCL', default=5e-2, type=float)
    parser.add_argument('--temprature', default=7e-2, type=float)
    parser.add_argument('--cent_lr', default=5e-1, type=float)
    parser.add_argument('--batch_size_train', default=512, type=int)
    parser.add_argument('--epochs', default=35, type=int)
    parser.add_argument('--num_features', default=256, type=int)
    parser.add_argument('--step1_decrease', default=25, type=int)
    parser.add_argument('--step2_decrease', default=30, type=int)
    parser.add_argument('--number_of_workers', default=10, type=int)
    parser.add_argument('--device', default='cuda', type=str)
   
    return parser
    


def main(args):
    now = dt.now()
    current_time = now.strftime("%d/%m/%Y\n%H:%M:%S") 
    checkpoint_time = now.strftime("_%d_%m_%Y_%H_%M_%S") 
    print('Current Date and Time:')
    print(current_time)
    
    print(args)
    start_time = time.time()    
    trainloader,testloader=GET_SSL_DB(args.batch_size_train,args.number_of_workers)
    print()
    print('Dataset has been loaded')
    

    model = WideResNet(args.depth,args.GT_classes,args.widen_factor,args.dropout)
    model = model.to(args.device)   
    
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_SCE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.step1_decrease,args.step2_decrease], gamma=args.gamma)
    
    criterion_center_loss = AMTCL(device=args.device, num_classes=args.GT_classes, num_dim=args.num_features).cuda()
    optimizer_centloss = torch.optim.AdamW(criterion_center_loss.parameters(), lr=args.cent_lr, weight_decay=args.weight_decay_AMTCL)

    train_loss_history= []
    min_loss=1000
    min_loss_epochs=1
    for epoch in range(1, args.epochs+1):
        print()
        print('Epoch:', epoch)
        start_of_epoch = time.time()
        
        train_loss=train_epoch(model,epoch, optimizer, optimizer_centloss, criterion_center_loss, args.temprature, trainloader, args.GT_classes, args.device)
        
        scheduler.step()
        
                
        if train_loss<min_loss:
            min_loss=train_loss
            min_loss_epochs=epoch
            print('Epoch Number of Min Loss up to now is:', min_loss_epochs)
            print('Min Loss is:',min_loss)
            checkpoint_path = os.getcwd()+'/self_supervised_learning'+str(epoch)+checkpoint_time+'_checkpoint.pth'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'criterion_center':criterion_center_loss.state_dict(),
                'args': args,
            }, checkpoint_path)
            
        
        train_loss_history.append(train_loss)
        print('Train Loss: ',train_loss)
        print('Epoch time:', datetime.timedelta(seconds=time.time() - start_of_epoch), '(hrs/mins/secs)')              
         
     

    print('Execution time:', datetime.timedelta(seconds=time.time() - start_time), '(hrs/mins/secs)')

if __name__ == '__main__':
    print('Started')
    parser = argparse.ArgumentParser('Deep Anomaly Detection--- Stage 1--- Self-Supervision', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
