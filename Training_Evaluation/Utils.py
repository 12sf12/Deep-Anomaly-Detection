#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:32:57 2021

@author: sf00511
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:50:06 2021

@author: sf00511
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:28:47 2021

@author: sf00511
"""
from torchvision.utils import save_image
import matplotlib
from torchvision.utils import make_grid
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.ion()   # interactive mode


# do gradient clip 
def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            param.grad.data.clamp_(-grad_clip, grad_clip)


def plot_auc(loss_type,loss_history,loss_history_distance,num_epochs,class_idx,DB_name,output_dir,current_time,num_repeats, josef_k):
    plt.title(loss_type+'_'+DB_name+':'+'Class_idx='+ str(class_idx)+'_Till Epoch:'+str(num_epochs)+'_Run:'+str(num_repeats)+'_adp_k:'+str(josef_k))
    plt.xlabel("Epochs")
    plt.ylabel("Test AUC")
    plt.plot(range(1,num_epochs+1),loss_history,label="EntropyAUC")
    plt.plot(range(1,num_epochs+1),loss_history_distance,label="DistanceAUC")
    #plt.ylim((0,1.))
    #plt.xticks(np.arange(1, num_epochs+2, 1.0))
    plt.legend()
    #plt.show()
    plt.savefig(output_dir+'/'+loss_type+'_'+DB_name+':'+ '_Test_auc_plot_Class_idx='+ str(class_idx)+'_till_epochs:'+str(num_epochs)+'_adp_k:'+str(josef_k)+'_repeats:'+str(num_repeats)+current_time+'.png')
    plt.close()

def plot_loss(loss_type,loss_history,loss_history_test,num_epochs,class_idx,DB_name,output_dir,current_time,num_repeats, josef_k):
    plt.title(loss_type+'_'+DB_name+':'+'Cls_idx='+ str(class_idx)+'_Till_Epoch:'+str(num_epochs)+'_Run:'+str(num_repeats)+'_adp_k:'+str(josef_k))
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy VS. Center Loss")
    plt.plot(range(1,num_epochs+1),loss_history,label="CrossEntropy")
    plt.plot(range(1,num_epochs+1),loss_history_test,label="CenterLoss")

    #plt.ylim((0,1.))
    #plt.xticks(np.arange(1, num_epochs+2, 1.0))
    plt.legend()
    #plt.show()
    plt.savefig(output_dir+'/'+loss_type+'_'+DB_name+':'+ '_Cross_Center_loss_plot_Class_idx='+ str(class_idx)+'_till_epochs:'+str(num_epochs)+'_adp_k:'+str(josef_k)+'_repeats:'+str(num_repeats)+current_time+'.png')
    plt.close()
    
def plot_loss_test(loss_type,loss_history,num_epochs,class_idx,DB_name,output_dir,current_time,num_repeats):
    plt.title(loss_type+'_'+DB_name+':'+'Class_idx='+ str(class_idx)+'_Till Epoch:'+str(num_epochs)+'_Run:'+str(num_repeats))
    plt.xlabel("Test Epochs")
    plt.ylabel("Test loss")
    plt.plot(range(1,num_epochs+2),loss_history,label="Test Loss")
    #plt.ylim((0,1.))
    #plt.xticks(np.arange(1, num_epochs+2, 1.0))
    plt.legend()
    #plt.show()
    plt.savefig(output_dir+'/'+loss_type+'_'+DB_name+':'+ '_Test_loss_plot_Class_idx='+ str(class_idx)+'_till_epochs:'+str(num_epochs)+'_Run:'+str(num_repeats)+current_time+'.png')
    plt.close()
    
def test_image_reconstruction(net, device, num_epochs, class_idx, DB_name, testloader,output_dir,current_time,num_repeats):
    transformer_bank=[]
    tr_idx=0
    for i in range(0,2):
        for j in range(-1,2):
            for z in range(-1,2):
                for w in range(0,4):
                    transformer_bank.append([i,j,z,w,tr_idx])
                    tr_idx=tr_idx+1
                    
    for batch in testloader:
        img, _ = batch
        img = img.to(device)
        #img,_=return_augmented_batch(img,_) 
        [batch_size,channels, height, width]=img.shape
        img = img.view(batch_size,channels, height, width)
        outputs = net(img)
        outputs = outputs[1].view(batch_size,channels, height, width).cpu().data
        save_image(outputs, output_dir+'/'+DB_name+':'+ '_reconstruction_Class_idx='+ str(class_idx)+'_till_epochs:'+str(num_epochs)+'_Run:'+str(num_repeats)+current_time+'.png')
        save_image(img, output_dir+'/'+DB_name+':'+ '_original_Class_idx='+ str(class_idx)+'_till_epochs:'+str(num_epochs)+'_Run:'+str(num_repeats)+current_time+'.png')
        break
    
