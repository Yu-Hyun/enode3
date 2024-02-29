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
import torch.nn.utils.weight_norm as weight_norm
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

class Eulermodel(nn.Module):
    def __init__(self, model, C):
        super(Eulermodel,self).__init__()
        self.model = model
        self.relu = nn.ReLU()
        self.C = C
        self.g1 = nn.Parameter(torch.ones(C)).cuda()
        self.g2 = nn.Parameter(torch.ones(C)).cuda()
        self.conv_1  = weight_norm(nn.Conv2d(C, C, 3, 1, 1,bias=False))
        self.conv_1.weight_g = nn.Parameter(torch.ones(C,1,1,1)/math.sqrt(C))
        self.conv_1.weight_g.requires_grad=False
        self.conv0_1 = weight_norm(nn.Conv2d(C, C, 1, 1, 0,bias=False)) # k s p
        self.conv_2  = weight_norm(nn.Conv2d(C, C, 3, 1, 1,bias=False))
        self.conv_2.weight_g = nn.Parameter(torch.ones(C,1,1,1)/math.sqrt(C))
        self.conv_2.weight_g.requires_grad=False
        self.conv0_2 = weight_norm(nn.Conv2d(C, C, 1, 1, 0,bias=False)) # k s p
        self.x01 = 0
        self.x02 = 0
        self.t = 1
        self.steps = 1
    def forward(self, x):
        x = self.model.forward_1(x)
        
        
        fx01 = self.relu(self.conv0_1(x))
        for idx in range(self.steps):
            x = x +(self.t/self.steps)*(-F.conv_transpose2d(self.relu(self.conv_1(x)),  self.conv_1.weight, stride=1,padding=1,output_padding=0,bias=None) + fx01 + -0.12*x )
        x = x*self.g1[None,:,None,None]
        
        fx02 = self.relu(self.conv0_2(x))
        for idx in range(self.steps):
            x = x +(self.t/self.steps)*(-F.conv_transpose2d(self.relu(self.conv_2(x)),  self.conv_2.weight, stride=1,padding=1,output_padding=0,bias=None) + fx02 + -0.12*x )
        x = x*self.g2[None,:,None,None]
        x = self.relu(x)
        
        
        x = self.model.forward_NN(x)
        return x
    def forward_L1(self, x):
        x = self.model.forward_1(x)
        
        f_before = x
        
        
        fx01 = self.relu(self.conv0_1(x))
        for idx in range(self.steps):
            x = x +(self.t/self.steps)*(-F.conv_transpose2d(self.relu(self.conv_1(x)),  self.conv_1.weight, stride=1,padding=1,output_padding=0,bias=None) + fx01 + -0.12*x )
        x = x*self.g1[None,:,None,None]
        
        fx02 = self.relu(self.conv0_2(x))
        for idx in range(self.steps):
            x = x +(self.t/self.steps)*(-F.conv_transpose2d(self.relu(self.conv_2(x)),  self.conv_2.weight, stride=1,padding=1,output_padding=0,bias=None) + fx02 + -0.12*x )
        x = x*self.g2[None,:,None,None]
        x = self.relu(x)
        
        
        f_after = x
        
        x = self.model.forward_NN(x)
        return x, f_before, f_after
        
    def copy_euler_model(self, euler_net):
        self.g1 = nn.Parameter(euler_net.g1.clone().detach())
        self.g2 = nn.Parameter(euler_net.g2.clone().detach())
        self.conv_1.weight  = euler_net.conv_1.weight  
        self.conv_1.weight_g = nn.Parameter(euler_net.conv_1.weight_g.clone().detach())
        self.conv_1.weight_v = nn.Parameter(euler_net.conv_1.weight_v.clone().detach())
        self.conv0_1.weight_g = nn.Parameter(euler_net.conv0_1.weight_g.clone().detach())
        self.conv0_1.weight_v = nn.Parameter(euler_net.conv0_1.weight_v.clone().detach())
        self.conv0_1.weight = euler_net.conv0_1.weight
        
        self.conv_2.weight  = euler_net.conv_2.weight
        self.conv_2.weight_g = nn.Parameter(euler_net.conv_2.weight_g.clone().detach())
        self.conv_2.weight_v = nn.Parameter(euler_net.conv_2.weight_v.clone().detach())
        self.conv0_2.weight_g = nn.Parameter(euler_net.conv0_2.weight_g .clone().detach())
        self.conv0_2.weight_v = nn.Parameter(euler_net.conv0_2.weight_v.clone().detach())
        self.conv0_2.weight = euler_net.conv0_2.weight
        
        
def train(epoch):
    start = time.time()
    net.train()
    net.model.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            
            adv_images = atk_train(images, labels).cuda()
            optimizer.zero_grad()
        if args.lmbd == 0 :
            outputs = net(adv_images)
            loss = loss_function(outputs, labels)
        elif args.lmbd == 0.01 : 
            B,_,_,_ = images.size()
            images = torch.cat( (images,adv_images) ,dim=0)
            outputs, f_before_ode, f_after_ode = net.forward_L1(images)
            loss_t = loss_function(outputs[B:2*B,:], labels)
            
            _,c,h,w = f_before_ode.size()
            f_clean = torch.split(f_before_ode,B,dim=0)[0].cuda()
            f_after_ode_split = torch.split(f_after_ode,B,dim=0)
            f_dist = f_after_ode_split[0].cuda()
            f_denoised = f_after_ode_split[1].cuda()
            
            dist_loss =  torch.sum( torch.abs(f_clean - f_dist), dim=(0,1,2,3)) + torch.sum( torch.abs(f_clean - f_denoised), dim=(0,1,2,3))
            loss = loss_t + dist_loss * args.lmbd/(B*c*h*w)
        
        
        loss.backward()
        optimizer.step()
        
        if epoch <= args.warm:
            warmup_scheduler.step()
            
    finish = time.time()
    logger.info('training epoch {} ended. Time consumed : {:.2f}s'.format(epoch,finish - start))
            
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
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-lmbd', type=float, default=0.01, help='distortion loss hyper parameter')
    parser.add_argument('-r', type=eval, default=True, choices=[True, False], help = 'set True to repeatedly breached scenario')
    args = parser.parse_args()
    '''
    net = get_network(args)
    '''
    
    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_cifar100/checkpoint/vgg16/Wednesday_13_September_2023_17h_28m_20s/vgg16-200-regular.pth'))
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_cifar100/checkpoint/resnet18/Wednesday_13_September_2023_18h_06m_23s/resnet18-200-regular.pth'))
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_cifar100/checkpoint/inceptionv3/Wednesday_13_September_2023_18h_11m_57s/inceptionv3-200-regular.pth'))
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_cifar100/checkpoint/wideresnet/Wednesday_13_September_2023_18h_14m_14s/wideresnet-200-regular.pth'))
    elif args.net == 'inceptionv4' : 
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_cifar100/checkpoint/inceptionv4/Wednesday_13_September_2023_18h_12m_29s/inceptionv4-200-regular.pth'))
        
    net = net.eval().cuda()
    for param in net.parameters():
        param.requires_grad = False
    net = net.cuda()
    
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader( # actually this is cifar10 loader not 100 loader 
        settings.CIFAR100_TRAIN_MEAN, 
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    if args.net == 'resnet18' : 
        ODE_channels = 64

    if args.net == 'vgg16' : 
        ODE_channels = 64
        
    if args.net == 'inceptionv3' : 
        ODE_channels = 32#192
    
    if args.net == 'inceptionv4' : 
        ODE_channels = 384 # 192 + 192
        
    if args.net == 'wideresnet' : 
        ODE_channels = 16#160
        
    net = Eulermodel(model=net, C=ODE_channels).cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    net_fixed = net.model.eval()

    atk_train = pgd(model = net_fixed)
    atk_train.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 
    
    atk_fgsm_test = fgsm(model = net_fixed.eval())
    atk_fgsm_test.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 
    
    atk_pgd_test = pgd(model = net_fixed.eval(), steps=20)
    atk_pgd_test.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 
    
    atk_cw_test = CW(model = net_fixed.eval())
    atk_cw_test.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 

    atk_mifgsm_test = MIFGSM(model = net_fixed.eval())
    atk_mifgsm_test.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 
    
    atk_tifgsm_test = TIFGSM(model = net_fixed.eval())
    atk_tifgsm_test.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 

    atk_difgsm_test = DIFGSM(model = net_fixed.eval())
    atk_difgsm_test.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 
    
    atk_nifgsm_test = NIFGSM(model = net_fixed.eval())
    atk_nifgsm_test.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 

    if args.r == True : 
        if args.lr == 0.1 :
            if args.lmbd == 0.01 : 
                checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/rODE_dist_loss', settings.TIME_NOW)
            elif args.lmbd == 0.0 : 
                checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/rODE_loss_t', settings.TIME_NOW)
        elif args.lr == 0.01 : 
            if args.lmbd == 0.01 : 
                checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/rODE_FT_dist_loss', settings.TIME_NOW)
            elif args.lmbd == 0.0 : 
                checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/rODE_FT_loss_t', settings.TIME_NOW)
    if args.r == False : 
        if args.lr == 0.1 :
            if args.lmbd == 0.01 : 
                checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/ODE_dist_loss', settings.TIME_NOW)
            elif args.lmbd == 0.0 : 
                checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/ODE_loss_t', settings.TIME_NOW)
        elif args.lr == 0.01 : 
            if args.lmbd == 0.01 : 
                checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/ODE_FT_dist_loss', settings.TIME_NOW)
            elif args.lmbd == 0.0 : 
                checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/ODE_FT_loss_t', settings.TIME_NOW)

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
        acc = eval_training(epoch)
        
        if args.r == True :
            if epoch >2 and epoch % 3 == 0 :
                net_fixed = Eulermodel(model=net.model, C=ODE_channels).cuda()
                net_fixed.copy_euler_model(net)
                for param in net_fixed.parameters():
                    param.requires_grad = False
                    
                net_fixed = net_fixed.cuda()
                ###############
                net = Eulermodel(model=net.model, C=ODE_channels).cuda() # initialize ODE moduel
                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                ################
                atk_train = pgd(model = net_fixed.eval())
                atk_train.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 
                
                atk_fgsm_test = fgsm(model = net_fixed.eval())
                atk_fgsm_test.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 
                
                atk_pgd_test = pgd(model = net_fixed.eval(), steps=20)
                atk_pgd_test.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 
                
                atk_cw_test = CW(model = net_fixed.eval())
                atk_cw_test.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 
                
                atk_mifgsm_test = MIFGSM(model = net_fixed.eval())
                atk_mifgsm_test.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 
                
                atk_tifgsm_test = TIFGSM(model = net_fixed.eval())
                atk_tifgsm_test.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 
                
                atk_difgsm_test = DIFGSM(model = net_fixed.eval())
                atk_difgsm_test.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 
                
                atk_nifgsm_test = NIFGSM(model = net_fixed.eval())
                atk_nifgsm_test.set_normalization_used(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404 ]) # cifar100 
                
                eval_training(epoch = epoch, before = True)

        if not epoch % 3:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            logger.info('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
    
    