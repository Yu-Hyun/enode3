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

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
    
####
import torch.nn.functional as F
import torch.nn as nn
import math

from torchattacks import FGSM as fgsm
from torchattacks import PGD as pgd
from torchattacks import CW as CW
from torchattacks import MIFGSM as MIFGSM
from torchattacks import TIFGSM as TIFGSM
from torchattacks import DIFGSM as DIFGSM
from torchattacks import NIFGSM as NIFGSM

import logging

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger 
        
def train(epoch):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            adv_images = atk_pgd_train(images, labels).cuda()
            
        optimizer.zero_grad()
        outputs = net(adv_images)
        
        loss_t = loss_function(outputs, labels)
        
        loss = loss_t 
        
        loss.backward()
        optimizer.step()

        if epoch <= args.warm:
            warmup_scheduler.step()
            
    finish = time.time()
    print('training epoch {} ended. Time consumed : {:.2f}s'.format(epoch,finish - start))

#@torch.no_grad() #grad is needed for calculating adversarial examples
def eval_training(epoch=0, before = False):
    start = time.time()
    net.eval()
    correct = 0.0
    correct_fgsm = 0.0
    correct_pgd = 0.0
    correct_cw = 0.0
    correct_mifgsm = 0.0
    correct_tifgsm = 0.0
    correct_difgsm = 0.0
    correct_nifgsm = 0.0

    for (images, labels) in cifar100_test_loader:

        images = images.cuda()
        labels = labels.cuda()
        
        img_fgsm = atk_fgsm_test(images, labels).cuda()
        
        img_pgd = atk_pgd_test(images, labels).cuda()
        
        img_cw = atk_cw_test(images, labels).cuda()
        
        img_mifgsm = atk_mifgsm_test(images, labels).cuda()
        
        img_tifgsm = atk_tifgsm_test(images, labels).cuda()
    
        img_difgsm = atk_difgsm_test(images, labels).cuda()
        
        img_nifgsm = atk_nifgsm_test(images, labels).cuda()
        
        
        outputs = net(images).cuda()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        
        outputs = net(img_fgsm).cuda()
        _, preds = outputs.max(1)
        correct_fgsm += preds.eq(labels).sum()
        
        outputs = net(img_pgd).cuda()
        _, preds = outputs.max(1)
        correct_pgd += preds.eq(labels).sum()
        
        outputs = net(img_cw).cuda()
        _, preds = outputs.max(1)
        correct_cw += preds.eq(labels).sum()
        
        outputs = net(img_mifgsm).cuda()
        _, preds = outputs.max(1)
        correct_mifgsm += preds.eq(labels).sum()
        
        outputs = net(img_tifgsm).cuda()
        _, preds = outputs.max(1)
        correct_tifgsm += preds.eq(labels).sum()
        
        outputs = net(img_difgsm).cuda()
        _, preds = outputs.max(1)
        correct_difgsm += preds.eq(labels).sum()
        
        outputs = net(img_nifgsm).cuda()
        _, preds = outputs.max(1)
        correct_nifgsm += preds.eq(labels).sum()
        
            
            
        
    finish = time.time()
    if before == True : 
        logger.info('evaluaing right after breaching. . .')
    logger.info('Epoch: {},  Acc: {:.4f}, fgsm : {:.4f}, pgd : {:.4f}, cw : {:.4f}, mifgsm : {:.4f}, tifgsm : {:.4f}, difgsm : {:.4f}, nifgsm : {:.4f} Time consumed:{:.2f}s'.format(
        epoch,
        correct.float() / len(cifar100_test_loader.dataset),
        correct_fgsm.float() / len(cifar100_test_loader.dataset),
        correct_pgd.float() / len(cifar100_test_loader.dataset),
        correct_cw.float() / len(cifar100_test_loader.dataset),
        correct_mifgsm.float() / len(cifar100_test_loader.dataset),
        correct_tifgsm.float() / len(cifar100_test_loader.dataset),
        correct_difgsm.float() / len(cifar100_test_loader.dataset),
        correct_nifgsm.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    
    return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate', choices=[0.1, 0.01])
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-r', type=eval, default=True, choices=[True, False], help = 'set True to repeatedly breached scenario')
    args = parser.parse_args()
    
    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_cifar10/checkpoint/vgg16/cifar10_Thursday_14_September_2023_15h_12m_59s/vgg16-200-regular.pth'))
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_cifar10/checkpoint/resnet18/cifar10_Thursday_14_September_2023_15h_12m_19s/resnet18-200-regular.pth'))
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_cifar10/checkpoint/inceptionv3/cifar10_Thursday_14_September_2023_15h_14m_39s/inceptionv3-200-regular.pth'))
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_cifar10/checkpoint/wideresnet/cifar10_Thursday_14_September_2023_18h_34m_10s/wideresnet-200-regular.pth'))
    elif args.net == 'inceptionv4' : 
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_cifar10/checkpoint/inceptionv4/cifar10_Thursday_14_September_2023_15h_28m_51s/inceptionv4-200-regular.pth'))
        
    net = net.eval().cuda()
    for param in net.parameters():
        param.requires_grad = True
    net = net.cuda()
    
    import copy
    net_fixed = copy.deepcopy(net)
    for param in net_fixed.parameters():
        param.requires_grad = False
    net_fixed = net_fixed #.to('cuda:1')
    
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
    
    atk_pgd_train = pgd(model = net)
    atk_pgd_train.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    
    atk_fgsm_test = fgsm(model = net_fixed.eval())
    atk_fgsm_test.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    
    atk_pgd_test = pgd(model = net_fixed.eval(), steps=20)
    atk_pgd_test.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]) 
    
    atk_cw_test = CW(model = net_fixed.eval())
    atk_cw_test.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    
    atk_mifgsm_test = MIFGSM(model = net_fixed.eval())
    atk_mifgsm_test.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]) 
    
    atk_tifgsm_test = TIFGSM(model = net_fixed.eval())
    atk_tifgsm_test.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]) 

    atk_difgsm_test = DIFGSM(model = net_fixed.eval())
    atk_difgsm_test.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]) 
    
    atk_nifgsm_test = NIFGSM(model = net_fixed.eval())
    atk_nifgsm_test.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]) 


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.r == True : 
        if args.lr == 0.1 :
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/rAT', settings.TIME_NOW)
        elif args.lr == 0.01 : 
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/rAFT', settings.TIME_NOW)
    if args.r == False : 
        if args.lr == 0.1 :
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/AT', settings.TIME_NOW)
        elif args.lr == 0.01 : 
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/AFT', settings.TIME_NOW)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    logger = get_logger(logpath=os.path.join(checkpoint_path, 'logs'), filepath=os.path.abspath(__file__))
    logger.propagate = False
    logger.info(args)  
    
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    for epoch in range(1, 30 + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)
            
        train(epoch)
        eval_training(epoch)
        
        if args.r == True : 
            if epoch >2 and epoch % 3 == 0 :
                net_fixed = copy.deepcopy(net)
                
                for param in net_fixed.parameters():
                    param.requires_grad = False
                net_fixed = net_fixed.cuda()
                
                atk_fgsm_test = fgsm(model = net_fixed.eval())
                atk_fgsm_test.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]) 
                
                atk_pgd_test = pgd(model = net_fixed.eval(), steps=20)
                atk_pgd_test.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]) 
                
                atk_cw_test = CW(model = net_fixed.eval())
                atk_cw_test.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]) 
            
                atk_mifgsm_test = MIFGSM(model = net_fixed.eval())
                atk_mifgsm_test.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]) 
                
                atk_tifgsm_test = TIFGSM(model = net_fixed.eval())
                atk_tifgsm_test.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]) 
            
                atk_difgsm_test = DIFGSM(model = net_fixed.eval())
                atk_difgsm_test.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]) 
                
                atk_nifgsm_test = NIFGSM(model = net_fixed.eval())
                atk_nifgsm_test.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]) 
                
                eval_training(epoch = epoch, before = True)

        if not epoch % 3:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            logger.info('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
    