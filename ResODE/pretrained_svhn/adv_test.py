# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
    
####
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
import torch.nn as nn
import math

from torchattacks import FGSM as fgsm
from torchattacks import PGD as pgd


def train(epoch):

    start = time.time()
    max_svhn = -math.inf
    min_svhn = math.inf
    mean=torch.tensor([0.4376821, 0.4437697, 0.47280442]).cuda()
    std=torch.tensor([0.19803012, 0.20101562, 0.19703614]).cuda()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            B,_,_,_ = images.size()
            
            un_normalized_img = images*std[None,:,None,None] + mean[None,:,None,None]
            max_val = torch.max(torch.max(torch.max(torch.max(un_normalized_img)))) 
            min_val = torch.min(torch.min(torch.min(torch.min(un_normalized_img))))
            if max_val < max_svhn :
                max_svhn = max_val
            if min_val > min_svhn :
                min_val = min_svhn
            
            
    finish = time.time()

    print('min val : ',min_val,' max val : ', max_val)

#@torch.no_grad() #grad is needed for calculating adversarial examples
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()
    
    test_loss = 0.0 # cost function error
    correct = 0.0
    correct_fgsm = 0.0
    correct_pgd = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images).cuda()
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        
    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, fgsm Acc : {:.4f}, pgd Acc : {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        -1.0,
        -1.0,
        finish - start
    ))
    
    print() 
    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-tol', type=float, default=1e-3, help = 'default ODE solver tolerance')
    args = parser.parse_args()
    '''
    net = get_network(args)
    '''
    
    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_svhn/checkpoint/vgg16/SVHN_Thursday_14_September_2023_09h_29m_31s/vgg16-200-regular.pth'))
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_svhn/checkpoint/resnet18/SVHN_Wednesday_13_September_2023_18h_57m_00s/resnet18-200-regular.pth'))
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_svhn/checkpoint/inceptionv3/SVHN_Thursday_14_September_2023_09h_31m_07s/inceptionv3-200-regular.pth'))
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_svhn/checkpoint/inceptionv4/SVHN_Thursday_14_September_2023_09h_31m_51s/inceptionv4-200-regular.pth'))
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_svhn/checkpoint/inceptionresnetv2/SVHN_Thursday_14_September_2023_09h_30m_21s/inceptionresnetv2-200-regular.pth'))
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_svhn/checkpoint/wideresnet/SVHN_Thursday_14_September_2023_09h_32m_17s/wideresnet-200-regular.pth'))
        
    net = net.eval().cuda()
    for param in net.parameters():
        param.requires_grad = True
    net = net.cuda()
    
    import copy
    net_fixed = copy.deepcopy(net)
    for param in net_fixed.parameters():
        param.requires_grad = False
    net_fixed = net_fixed.cuda()

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader( # actually svhn loader
        settings.SVHN_TRAIN_MEAN, 
        settings.SVHN_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader( # actually svhn loader
        settings.SVHN_TRAIN_MEAN,
        settings.SVHN_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    
    atk_fgsm_train = fgsm(model = net)
    atk_fgsm_train.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue
            
        train(epoch)
        acc = eval_training(epoch) 
    