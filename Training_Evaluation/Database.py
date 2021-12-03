#DEEP ANOMALY DETECTION by SF12


import torchvision.transforms as transforms
import torchvision
import torch
import numpy as np
from torch.utils.data import Subset 
from torchvision.transforms import functional as tvF
from numpy.random import randint


def modify_DB_for_class(DB_name, class_idx, BATCH_SIZE_VAL,NUMBER_OF_WORKERS):
        

    transform = transforms.Compose(        
    [transforms.Resize([32,32]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])      
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                    download=True, transform=transform)
    
    trainset.data=np.expand_dims(trainset.data, axis=-1)
    trainset.data=np.repeat(trainset.data,3,3)
    
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                   download=True, transform=transform)
    
    testset.data=np.expand_dims(testset.data, axis=-1)
    testset.data=np.repeat(testset.data,3,3)
        
 
    train_targets = trainset.targets   
    target_indices = np.arange(len(train_targets))
    idx_to_keep = train_targets[target_indices]==class_idx
    train_idx = target_indices[idx_to_keep]
    trainset_modified = Subset(trainset, train_idx)
    
    test_targets = testset.targets
    test_indices = np.arange(len(test_targets))
    idx_target_class = test_targets[test_indices]==class_idx
    idx_opposite_classes = test_targets[test_indices]!=class_idx

        
    testset.targets[idx_target_class]=0
    testset.targets[idx_opposite_classes]=1
    sf_pc=10
    print('how much of test db is used (percentage): ', 100/sf_pc)
    evens = list(randint(0, len(testset),int(len(testset)/sf_pc)))
    testset = torch.utils.data.Subset(testset, evens)  
      
    trainloader = torch.utils.data.DataLoader(trainset_modified, batch_size=BATCH_SIZE_VAL,
                                              shuffle=True, num_workers=NUMBER_OF_WORKERS)

    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE_VAL,
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
    


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]
   
