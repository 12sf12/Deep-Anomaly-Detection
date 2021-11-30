#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:28:47 2021

@author: sf00511
"""
import numpy as np
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()   # interactive mode

def plot_loss(loss_history,num_epochs,class_idx,DB_name,output_dir,current_time,num_repeats):
    plt.title(DB_name+': '+'Class_idx='+ str(class_idx)+ ' Till Epoch: '+str(num_epochs)+'_Run:'+str(num_repeats))
    plt.xlabel("Training Epochs")
    plt.ylabel("Training loss")
    plt.plot(range(1,num_epochs+1),loss_history,label="Training Loss")
    plt.ylim((0,2.))
    plt.xticks(np.arange(1, num_epochs+1, 50.0))
    plt.legend()
    plt.savefig(output_dir+'/'+DB_name+': '+ '_loss_plot_Class_idx= '+ str(class_idx)+'_till_epochs:'+str(num_epochs)+'_Run:'+str(num_repeats)+current_time+'.png')
    plt.close()
    
def test_image_reconstruction(net, device, num_epochs, class_idx, DB_name, testloader,output_dir,current_time,num_repeats):
     for batch in testloader:
        img, _ = batch
        img = img.to(device)
        [batch_size,channels, height, width]=img.shape
        img = img.view(batch_size,channels, height, width)
        outputs = net(img)
        outputs = outputs[0].view(batch_size,channels, height, width).cpu().data
        save_image(outputs, output_dir+'/'+DB_name+': '+ '_reconstruction_Class_idx='+ str(class_idx)+'_till_epochs:'+str(num_epochs)+'_Run:'+str(num_repeats)+current_time+'.png')
        save_image(img, output_dir+'/'+DB_name+': '+ '_original_Class_idx='+ str(class_idx)+'_till_epochs:'+str(num_epochs)+'_Run:'+str(num_repeats)+current_time+'.png')
        break