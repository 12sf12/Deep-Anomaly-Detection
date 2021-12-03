#DEEP ANOMALY DETECTION by SF12

import torch
import time
import torch.optim as optim
import argparse
from Database import modify_DB_for_class
from Execution import train_epoch, evaluate, evaluate_means
from datetime import datetime

from wideresnet import WideResNet
from Triplet_center_loss import AMTCL
#import utils

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

torch.manual_seed(1371)
def get_args_parser():
    parser = argparse.ArgumentParser('Deep Anomaly Detection', add_help=False)
    parser.add_argument('--db_name', default='Fashion_MNIST', type=str)
    parser.add_argument('--model', default='wideresnet', type=str)
    parser.add_argument('--class_id', default=1, type=int)
    parser.add_argument('--GT_classes', default=8, type=int)
    parser.add_argument('--depth', default=40, type=int)
    parser.add_argument('--widen_factor', default=4, type=int)
    parser.add_argument('--dropout', default=3e-1, type=float)
    parser.add_argument('--step1_decrease', default=25, type=int)
    parser.add_argument('--step2_decrease', default=30, type=int)
    parser.add_argument('--ce_flag', default=1, type=int)
    parser.add_argument('--centre_flag', default=1, type=int)
    parser.add_argument('--gamma', default=0.3, type=float)
    parser.add_argument('--weight_decay_SCE', default=5e-3, type=float)
    parser.add_argument('--weight_decay_AMTCL', default=5e-2, type=float)
    parser.add_argument('--knn_param', default=5, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--cent_lr', default=1e-1, type=float)
    parser.add_argument('--temprature', default=7e-2, type=float)
    parser.add_argument('--batch_size_train', default=256, type=int)
    parser.add_argument('--repeats', default=5, type=int)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--num_features', default=256, type=int)
    parser.add_argument('--k_val', default=3, type=int)
    parser.add_argument('--number_of_workers', default=20, type=int)
    parser.add_argument('--eval_per_epochs', default=1, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--ssl_address', default='', type=str)
   
    return parser
    

def main(args):
    now = datetime.now()
    current_time = now.strftime("%d/%m/%Y\n%H:%M:%S") 
    print('Current Date and Time:')
    print(current_time)
    
    print(args)
    start_time = time.time()
    
    torch.cuda.is_available()
    torch.zeros(1).cuda()

    trainloader,testloader=modify_DB_for_class(args.db_name,args.class_id,args.batch_size_train,args.number_of_workers)
   
    for num_repeats in range(args.repeats):
        print('**********************************')
        print('Run number:', num_repeats+1)
        print()
        

        model = WideResNet(args.depth,args.GT_classes,args.widen_factor,args.dropout)                   
        checkpoint=torch.load(args.ssl_address)
        model.load_state_dict(checkpoint['model'])
        model = model.to(args.device)


        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_SCE)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.step1_decrease,args.step2_decrease], gamma=args.gamma)
        
        criterion_center_loss = AMTCL(device=args.device, num_classes=args.GT_classes, num_dim=args.num_features).cuda()
        optimizer_centloss = torch.optim.AdamW(criterion_center_loss.parameters(), lr=args.cent_lr, weight_decay=args.weight_decay_AMTCL)
        
    
        train_loss_history= []
        min_loss=1000
        min_loss_epochs=1
        current_auc=[]
        for epoch in range(1, args.epochs+1):
            print()
            print('Epoch:', epoch)
            print('class_To_Check:', args.class_id)
            
            train_loss, cross_loss, center_loss=train_epoch(model,epoch, optimizer, optimizer_centloss, criterion_center_loss, args.temprature, trainloader, args.ce_flag, args.centre_flag, args.GT_classes, args.device)
            means=evaluate_means(model, trainloader, args.temprature, args.GT_classes , args.device)
            
            scheduler.step()

            
            if epoch%args.eval_per_epochs==0:
                current_auc.append(evaluate(model, testloader, trainloader, criterion_center_loss.centers, criterion_center_loss.centers_weights, means,  args.knn_param, args.temprature, args.GT_classes, args.device))
                print('EPOCH AUC is: ', current_auc[epoch-1])

                
            if train_loss<min_loss:
                min_loss=train_loss
                min_loss_epochs=epoch
                print('Epoch Number of Min Loss up to now is:', min_loss_epochs)
                print('Min Loss is:',min_loss)
            
            train_loss_history.append(train_loss)
            print('Train Loss: ',train_loss)
            
                        
    
    
    print('Execution time:', datetime.timedelta(seconds=time.time() - start_time), '(hrs/mins/secs)')



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deep Anomaly Detection', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
