#DEEP ANOMALY DETECTION by SF12

import torchvision.transforms as transforms
import torchvision
import torch
import numpy as np
from torch.utils.data import Subset 
#from sklearn.metrics import multilabel_confusion_matrix
#from vision_transformer_pytorch import VisionTransformer
from random import seed
from random import random, randint
import kornia
from kornia.constants import Resample
from kornia.color import *
from kornia import augmentation as K
import kornia.augmentation.random_generator as rg
from torchvision.transforms import functional as tvF
from torchvision.transforms import transforms
import torch.nn as nn
from torchvision import datasets, models, transforms, utils
from vit_pytorch import *
import time
import os
import pandas as pd
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
#from skimage import io
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import copy
import csv
import argparse
from datetime import datetime
import matplotlib
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision import datasets, models, transforms, utils
import torch.nn as nn
import torch.optim as optim
from TinyImageNet import TinyImageNetDataset
from numpy.random import randint


def GET_SSL_DB(BATCH_SIZE_VAL,NUMBER_OF_WORKERS):
    
    transform_images = transforms.Compose(
    [transforms.Resize([32,32]),
    #transforms.Resize([256,256]),
    transforms.CenterCrop([32,32]),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    
    root_dir=os.getcwd()+'/TinyImageNet/'
    mode='train'
    dataset_train = TinyImageNetDataset(root_dir=root_dir, mode=mode, transform=transform_images)
    
    mode='val'   
    dataset_val = TinyImageNetDataset(root_dir=root_dir, mode=mode, transform=transform_images)
    
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE_VAL,
                                          shuffle=True, num_workers=NUMBER_OF_WORKERS)

    testloader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE_VAL,
                                         shuffle=True, num_workers=NUMBER_OF_WORKERS)
    
    return trainloader,testloader

def return_augmented_test(images,targets,transformer_sf,t_idx):
        i=0
        [len_batch,C,height,width]=images.size()
        
        for i in range(int(len_batch)):
            if transformer_sf[t_idx][0]==0:
                images[i]=tvF.affine(images[i], angle=transformer_sf[t_idx][1]*90, translate=[0,0], scale=1, shear=0) 
                targets[i]=t_idx
            else:
                images[i]=tvF.affine(tvF.hflip(images[i]), angle=transformer_sf[t_idx][1]*90, translate=[0,0], scale=1, shear=0)
                targets[i]=t_idx                    
                                                                 
        return images,targets
    
    
def return_augmented_train(images,targets,transformer_sf,t_idx):
        i=0
        [len_batch,C,height,width]=images.size()
        
        for i in range(int(len_batch)):
            if transformer_sf[t_idx[i]][0]==0:
                images[i]=tvF.affine(images[i], angle=transformer_sf[t_idx[i]][1]*90, translate=[0,0], scale=1, shear=0) 
                targets[i]=t_idx[i]
            else:
                images[i]=tvF.affine(tvF.hflip(images[i]), angle=transformer_sf[t_idx[i]][1]*90, translate=[0,0], scale=1, shear=0)
                targets[i]=t_idx[i]                    
                                                                 
        return images,targets
   
