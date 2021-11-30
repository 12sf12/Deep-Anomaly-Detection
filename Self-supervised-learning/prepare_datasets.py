import os

from torchvision import transforms
import torch
import torchvision
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torchvision import datasets, transforms
from datasets.TinyImageNet import TinyImageNetDataset
from datasets.CIFAR import CIFAR10, CIFAR100
from datasets.STL10 import STL10
import numpy as np
from torch.utils.data import Subset 
from numpy.random import randint


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)


    mode='train' if is_train else 'val'
    root_dir = os.path.join(args.dataset_location, 'TinyImageNet/tiny-imagenet-200/')
    dataset = TinyImageNetDataset(root_dir=root_dir, mode=mode, transform=transform)
    nb_classes = 127


    return dataset, nb_classes





def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


