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
import torch.nn.functional as F
import torch.nn as nn
import math


from torchattacks import FGSM as fgsm
from torchattacks import PGD as pgd
        
def train(epoch):

    start = time.time()
    
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            B,_,_,_ = images.size()
            
            max_val = torch.max(torch.max(torch.max(images))) 
            min_val = torch.min(torch.min(torch.min(images)))
            atk_fgsm.eps = (16/255)/(max_val - min_val)
            adv_images = atk_fgsm((images - min_val)/(max_val - min_val), labels).cuda()
            adv_images = adv_images*(max_val - min_val) + min_val
            adv_images = adv_images.cuda()
        optimizer.zero_grad()
        outputs = net(adv_images)
        
        loss_t = loss_function(outputs, labels)
        
        loss = loss_t 
        
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        if epoch <= args.warm:
            warmup_scheduler.step()

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    
    

#@torch.no_grad() #grad is needed for calculating adversarial examples
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()
    
    test_loss = 0.0 # cost function error
    correct = 0.0
    correct_fgsm = 0.0
    correct_white = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images).cuda()
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
    
    for (images, labels) in cifar100_test_loader:
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            
            max_val = torch.max(torch.max(torch.max(images)))
            min_val = torch.min(torch.min(torch.min(images)))
            
            atk_fgsm.eps = (16/255)/(max_val - min_val)
            
            images_fgsm = atk_fgsm_fixed((images - min_val)/(max_val - min_val), labels)
            images_fgsm = images_fgsm*(max_val - min_val) + min_val  
            
        outputs = net(images_fgsm.cuda()).cuda()
        outputs_fgsm = net(images_fgsm.cuda()).cuda()
        _, preds_fgsm = outputs_fgsm.max(1)
        correct_fgsm += preds_fgsm.eq(labels.cuda()).sum()
    
    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, transfer Acc : {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        correct_fgsm.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    
    print()

    #add informations to tensorboard
    '''
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)
    '''
    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate', choices=[0.1, 0.01])
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-tol', type=float, default=1e-3, help = 'default ODE solver tolerance')
    args = parser.parse_args()
    '''
    net = get_network(args)
    '''
    
    if args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_cifar10/checkpoint/resnet18/cifar10_Thursday_14_September_2023_15h_12m_19s/resnet18-200-regular.pth'))
                                      #/home/yhshin/ResODE/pretrained_cifar10/checkpoint/resnet18/cifar10_Thursday_14_September_2023_15h_12m_19s/resnet18-200-regular.pth
        
    net = net.eval().cuda()
    for param in net.parameters():
        param.requires_grad = True
    net = net.cuda()
    
    import copy
    net_fixed = copy.deepcopy(net)
    for param in net_fixed.parameters():
        param.requires_grad = False
    net_fixed = net_fixed #.to('cuda:1')
    net_fixed = net_fixed.eval().cuda()
    
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader( # actually svhn loader
        settings.CIFAR10_TRAIN_MEAN, 
        settings.CIFAR10_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader( # actually svhn loader
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    
    atk_fgsm = fgsm(model= net_fixed)
    atk_fgsm.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    
    
    atk_fgsm_fixed = fgsm(model = net_fixed)
    atk_fgsm_fixed.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    acc = eval_training(-1)

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue
            
        train(epoch)
        acc = eval_training(epoch)
        
        # uncomment this area if you want to simulate repeated breach situation on AFT/AT
        if epoch % 5 == 0 and epoch > 4 : 
            net_fixed = copy.deepcopy(net.eval())
            for param in net_fixed.parameters():
                param.requires_grad = False
            net_fixed = net_fixed.eval().cuda()
                
            atk_fgsm = fgsm(model = net_fixed.eval())
            atk_fgsm.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
             
            atk_fgsm_fixed = fgsm(model = net_fixed.eval())
            atk_fgsm_fixed.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
        