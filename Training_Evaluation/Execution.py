#DEEP ANOMALY DETECTION by SF12

import torch
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
import math
from Database import return_augmented_test, return_augmented_train

    
     
def train_epoch(model, epoch, optimizer, optimizer_center, criterion_center, T, data_loader, ce_flag, centre_flag, GT_classes, device):
    
    if ce_flag !=0 or centre_flag !=0:
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
        print('W_center: ',w_center)
        print('W_cross: ',w_cross)
        for times in range(GT_classes):
            reach_id=0
            loss_times=0.0
            for data, target in data_loader:
                sample_len=len(data)
                orig_data=data
                orig_target=target
                start_idx=(times*data_len)+(sample_len*reach_id)
                end_idx=(times*data_len)+(sample_len*reach_id)+sample_len-1
                another_cpu_data,another_cpu_target=return_augmented_train(orig_data,orig_target,transformer_bank,idx_bank[start_idx:end_idx+1])
                reach_id+=1
                
                optimizer.zero_grad()
                optimizer_center.zero_grad()
                
                
                with torch.set_grad_enabled(True):
                    
                    features,logits = model(data.to(device))
                    if ce_flag:
                        term1 = F.cross_entropy(logits/T, target.to(device))
                    else:
                        term1=0
                    
                    if centre_flag:
                        term2= criterion_center(features.to(device), another_cpu_target.to(device),epoch)
                    else:
                        term2=0
                    
                    loss=term1*w_cross+term2*w_center        
                    loss.backward()
                    
                    optimizer.step()
                    if centre_flag:
                        optimizer_center.step()
                    
                total_loss+=loss*data.size(0)
                loss_times+=loss*data.size(0)
                if ce_flag:
                    total_term1+=term1*data.size(0)
                else:
                    total_term1+=0
                if centre_flag:    
                    total_term2+=term2*data.size(0)
                else:
                    total_term2+=0
                
        
        total_loss_train=total_loss.item()/(total_samples*GT_classes)
        print('Total Loss:' , total_loss.item()/(total_samples*GT_classes))
        if ce_flag:
            print('Loss Cross:' , total_term1.item()/(total_samples*GT_classes))
            loss_ce=total_term1.item()/(total_samples*GT_classes)
        else:
            loss_ce=0
        if centre_flag:
            print('Loss Center:' , total_term2.item()/(total_samples*GT_classes))
            loss_centre=total_term2.item()/(total_samples*GT_classes)
        else:
            loss_centre=0

    else:
        total_loss_train=-1
        print('No training')
        loss_ce=-1
        loss_centre=-1
        
    
    return total_loss_train, loss_ce, loss_centre
                
            
    
    
def evaluate_means(model, data_loader, T, GT_classes, device):
    model.eval()
    transformer_bank=[]
    arr = np.empty((8,0,8), float)
    tr_idx=0
    for i in range(0,2):
        for w in range(0,4):
            transformer_bank.append([i,w,tr_idx])
            tr_idx=tr_idx+1
                   
    with torch.no_grad():
        for data, target in data_loader:
            output_bank=np.zeros([GT_classes,len(target),GT_classes])
            orig_data=data
            orig_target=target
            for i in range(GT_classes):
                another_orig_data=orig_data
                another_orig_data,_= return_augmented_test(another_orig_data.to(device),orig_target.to(device),transformer_bank,i)  
                features,output=model(another_orig_data.to(device))
                output = F.softmax(output/T, dim=1)
                output_bank[i]+=output.tolist()
                
                
            arr = np.append(arr, np.array(output_bank.tolist()), axis=1)

    return np.mean(arr,1)


def gather_training_features(model, data_loader, device):
    model.eval()
    all_features=[]
    for data, target in data_loader:      
        with torch.no_grad():
            features,output=model(data.to(device))
            all_features.append(features.data.cpu().numpy())

    all_features = np.concatenate(all_features, 0)
    return all_features


def evaluate(model, data_loader, train_data_loader, centers, centers_weights, means, knn_val, T, GT_classes, device):
    
    model.eval()
    print()
    all_preds=[]
    all_preds_kl=[]
    all_preds3=[]
    all_labels=[]
    all_preds9=[]
    all_preds7=[]
    all_knns=[]
    transformer_bank=[]
    tr_idx=0
    for i in range(0,2):
        for w in range(0,4):
            transformer_bank.append([i,w,tr_idx])
            tr_idx=tr_idx+1
    
    training_features=gather_training_features(model, train_data_loader, device)               
    with torch.no_grad():
        for data, target in data_loader:
            output_bank9=np.zeros([len(target),GT_classes])
            output_bank=np.zeros(target.shape)
            output_bank2=np.zeros(target.shape)
            output_bank3=np.zeros(target.shape)
            output_bank7=np.zeros(target.shape)
            output_bank_KNN=np.zeros(target.shape)
            orig_data=data
            orig_target=target
            for i in range(GT_classes):
                another_orig_data=orig_data
                another_orig_target=orig_target
                another_orig_data,_= return_augmented_test(another_orig_data.to(device),another_orig_target.to(device),transformer_bank,i)  
                features,output=model(another_orig_data.to(device))
                
                
                output = F.softmax(output/T, dim=1)
                output=torch.clamp(output, min=1e-5)
                
                
                if i==0 :
                    for sf in range(len(features)):
                        output_bank_KNN[sf]=knn_eval(features[sf], training_features, knn_val)
                
                output_bank+=(-1*torch.sum(output*torch.log(output),1)).tolist()
                means[i]=np.clip(means[i], 1e-5, None)
                output_bank2+=np.sum(means[i]*np.log(means[i]/output.tolist()),1)
                output_bank3+=(-1*torch.sum(torch.log(output),1)).tolist()
                output_bank7+=np.sum(np.power((np.sqrt(means[i])-torch.sqrt(output).tolist()),2),1)
                
                
                for sf in range(len(features)):
                    output_bank9[sf][i]=math.sqrt(sum([(2**c) * ((a - b) ** 2) for a, b, c in zip(features[sf] , centers[i], centers_weights[i])]))

                
                
            
            all_labels=all_labels+(orig_target.tolist())
            pred=output_bank/GT_classes
            all_preds=all_preds+pred.tolist()
            
            pred9=np.mean(output_bank9,1)
            all_preds9=all_preds9+pred9.tolist()
                        
            
            pred_kl=output_bank2/GT_classes
            all_preds_kl=all_preds_kl+pred_kl.tolist()
            
            pred3=output_bank3/GT_classes
            all_preds3=all_preds3+pred3.tolist()  
            
            pred7=output_bank7/GT_classes
            all_preds7=all_preds7+pred7.tolist()
            
            all_knns=all_knns+output_bank_KNN.tolist()

    

    all_preds3=np.multiply(-1,all_preds3)  
    
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_preds, pos_label=1)
    auc1=metrics.auc(fpr, tpr)
    print('J1 = Entropy is:', auc1)
    
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_preds3, pos_label=1)
    auc2=metrics.auc(fpr, tpr)
    print('J2 = Softmax_Statistics is:', auc2)
    
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_preds_kl, pos_label=1)
    auc3=metrics.auc(fpr, tpr)
    print('J3 = KL_Divergence is:', auc3)
    
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_preds7, pos_label=1)
    auc4=metrics.auc(fpr, tpr)
    print('J4 = Squared Hellinger Distance is:', auc4)
    
    fpr, tpr, _ = metrics.roc_curve(all_labels, all_preds9, pos_label=1)
    auc2=metrics.auc(fpr, tpr)
    print('J5 = Distance to centres is:', auc2)
    
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_knns, pos_label=1)
    auc6=metrics.auc(fpr, tpr)
    print('J6 = KNN is:', auc6)    

    
    
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, two_sided_normalisation([all_preds, all_preds3, all_preds_kl, all_preds7, all_preds9, all_knns],1), pos_label=1)
    aucTotal=metrics.auc(fpr, tpr)
    
    
    return aucTotal

def knn_eval(feature, training_features, k):
    #distances=np.sqrt((sum([(a - b) ** 2 for a, b in zip(feature.data.cpu().numpy() ,training_features)])))
    distances=np.sqrt(np.sum([(feature.data.cpu().numpy() - b) ** 2 for b in training_features],1))
    distances=np.sort(distances)
    return np.average(distances[0:k])



def two_sided_normalisation(x,n):
    len_test=np.int(np.size(x)/6)
    output_bank=np.zeros(len_test)
    for i in range (6):        
        sf_min=np.percentile(x[i],n)
        sf_max=np.percentile(x[i],100-n)
        output_bank+=np.divide((np.subtract(x[i],sf_min)),(np.subtract(sf_max,sf_min)))
        
    return output_bank/6

