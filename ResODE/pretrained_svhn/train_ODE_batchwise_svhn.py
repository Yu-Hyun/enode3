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

import torchattacks
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
    
class Convmodel(nn.Module):
    def __init__(self, model, C):
        super(Convmodel,self).__init__()
        self.model = model
        self.relu = nn.ReLU()
        self.C = C
        self.conv_1  = nn.Conv2d(C, C, 3, 1, 1,bias=False)
        self.conv_2  = nn.Conv2d(C, C, 3, 1, 1,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=C)
        self.bn2 = nn.BatchNorm2d(num_features=C)
    def forward(self, x):
        x = self.model.forward_1(x)
        
        short_cut = x
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.bn2(x)
        
        x = self.relu(x + short_cut)
        
        x = self.model.forward_NN(x)
        return x
    def forward_L1(self, x):
        x = self.model.forward_1(x)
        
        f_before = x
        
        
        f_after = x
        
        x = self.model.forward_NN(x)
        return x, f_before, f_after
        
    def copy_conv_model(self, conv_net):
        self.load_state_dict(conv_net.state_dict())
        self.model = conv_net.model

class Eulermodel(nn.Module):
    def __init__(self, model, C):
        super(Eulermodel,self).__init__()
        self.model = model
        self.relu = nn.ReLU()
        self.C = C
        self.g1 = nn.Parameter(torch.ones(C)).cuda()
        self.g2 = nn.Parameter(torch.ones(C)).cuda()
        self.conv_1  = weight_norm(nn.Conv2d(C, C, 3, 1, 1,bias=False))
        self.conv_1.weight_g = nn.Parameter(args.gnorm*torch.ones(C,1,1,1)/math.sqrt(C))
        self.conv_1.weight_g.requires_grad=False
        self.conv0_1 = weight_norm(nn.Conv2d(C, C, 1, 1, 0,bias=False)) # k s p
        self.conv_2  = weight_norm(nn.Conv2d(C, C, 3, 1, 1,bias=False))
        self.conv_2.weight_g = nn.Parameter(args.gnorm*torch.ones(C,1,1,1)/math.sqrt(C))
        self.conv_2.weight_g.requires_grad=False
        self.conv0_2 = weight_norm(nn.Conv2d(C, C, 1, 1, 0,bias=False)) # k s p
        self.x01 = 0
        self.x02 = 0
        self.t = 1
        self.steps = 1
        
        self.n = 1
        
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
        
    def average_euler_model(self, euler_net):
        self.n = self.n + 1
        n = self.n
        
        alpha = float((n-1)/n)
        beta  = float(1/n)
        
        self.g1 = nn.Parameter(self.g1*alpha + beta*euler_net.g1.clone().detach())
        self.g2 = nn.Parameter(self.g2*alpha + beta*euler_net.g2.clone().detach())
        #self.conv_1.weight  = self.conv_1.weight*alpha + beta*euler_net.conv_1.weight.clone().detach()
        self.conv_1.weight_g = nn.Parameter(self.conv_1.weight_g*alpha + beta*euler_net.conv_1.weight_g.clone().detach())
        self.conv_1.weight_v = nn.Parameter(self.conv_1.weight_v*alpha + beta*euler_net.conv_1.weight_v.clone().detach())
        self.conv0_1.weight_g = nn.Parameter(self.conv0_1.weight_g*alpha + beta*euler_net.conv0_1.weight_g.clone().detach())
        self.conv0_1.weight_v = nn.Parameter(self.conv0_1.weight_v*alpha + beta*euler_net.conv0_1.weight_v.clone().detach())
        #self.conv0_1.weight = self.conv0_1.weight*alpha + beta*euler_net.conv0_1.weight.clone().detach()
        
        #self.conv_2.weight  = self.conv_2.weight*alpha + beta*euler_net.conv_2.weight
        self.conv_2.weight_g = nn.Parameter(self.conv_2.weight_g*alpha + beta*euler_net.conv_2.weight_g.clone().detach())
        self.conv_2.weight_v = nn.Parameter(self.conv_2.weight_v*alpha +beta*euler_net.conv_2.weight_v.clone().detach())
        self.conv0_2.weight_g = nn.Parameter(self.conv0_2.weight_g*alpha +beta*euler_net.conv0_2.weight_g .clone().detach())
        self.conv0_2.weight_v = nn.Parameter(self.conv0_2.weight_v*alpha +beta*euler_net.conv0_2.weight_v.clone().detach())
        #self.conv0_2.weight = self.conv0_2.weight*alpha + beta*euler_net.conv0_2.weight.clone().detach()
        
    def moving_average_euler_model(self, euler_net):
    
        alpha = 0.9
        beta = 0.1
        
        self.g1 = nn.Parameter(self.g1*alpha + beta*euler_net.g1.clone().detach())
        self.g2 = nn.Parameter(self.g2*alpha + beta*euler_net.g2.clone().detach())
        #self.conv_1.weight  = self.conv_1.weight*alpha + beta*euler_net.conv_1.weight.clone().detach()
        self.conv_1.weight_g = nn.Parameter(self.conv_1.weight_g*alpha + beta*euler_net.conv_1.weight_g.clone().detach())
        self.conv_1.weight_v = nn.Parameter(self.conv_1.weight_v*alpha + beta*euler_net.conv_1.weight_v.clone().detach())
        self.conv0_1.weight_g = nn.Parameter(self.conv0_1.weight_g*alpha + beta*euler_net.conv0_1.weight_g.clone().detach())
        self.conv0_1.weight_v = nn.Parameter(self.conv0_1.weight_v*alpha + beta*euler_net.conv0_1.weight_v.clone().detach())
        #self.conv0_1.weight = self.conv0_1.weight*alpha + beta*euler_net.conv0_1.weight.clone().detach()
        
        #self.conv_2.weight  = self.conv_2.weight*alpha + beta*euler_net.conv_2.weight.clone().detach()
        self.conv_2.weight_g = nn.Parameter(self.conv_2.weight_g*alpha + beta*euler_net.conv_2.weight_g.clone().detach())
        self.conv_2.weight_v = nn.Parameter(self.conv_2.weight_v*alpha +beta*euler_net.conv_2.weight_v.clone().detach())
        self.conv0_2.weight_g = nn.Parameter(self.conv0_2.weight_g*alpha +beta*euler_net.conv0_2.weight_g .clone().detach())
        self.conv0_2.weight_v = nn.Parameter(self.conv0_2.weight_v*alpha +beta*euler_net.conv0_2.weight_v.clone().detach())
        #self.conv0_2.weight = self.conv0_2.weight*alpha + beta*euler_net.conv0_2.weight.clone().detach()
        
def train(epoch):
    train_start = time.time()
    net.train()
    net.model.train()
    # svhn total batch numbers == 0~572 (573) for batch size 128
    for batch_index, (images, labels) in enumerate(cifar100_training_loader): 
        if batch_index % 57 == 0 or batch_index == 29 or batch_index == (57+29) : # every n*10% data learned, 5% data learned, and 15% data learned
            eval_training(batch = batch_index)
            optimizer.zero_grad()
            if epoch > 1 : 
                eval_training_avg(batch = batch_index)
            optimizer.zero_grad()
            
        net.train()
        net.model.train()
            
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            
            adv_images = atk_train(images, labels).cuda()
            optimizer.zero_grad()
        outputs = net(adv_images)
        loss = loss_function(outputs, labels)
        
        
        loss.backward()
        optimizer.step()
        
        trained_samples=batch_index * args.b + len(images),
        total_samples=len(cifar100_training_loader.dataset)
        
        
        if epoch <= args.warm:
            warmup_scheduler.step()
            
    train_finish = time.time()
    logger.info('training epoch {} ended. Time consumed : {:.2f}s'.format(epoch,train_finish - train_start))

def eval_training(batch = 0):

    eval_start = time.time()
    net.eval()
    net.model.eval()
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
            
        
    eval_finish = time.time()
    logger.info('batch : {}, Acc: {:.4f}, fgsm : {:.4f}, pgd : {:.4f}, cw : {:.4f}, mifgsm : {:.4f}, tifgsm : {:.4f}, difgsm : {:.4f}, nifgsm : {:.4f} Time consumed:{:.2f}s'.format(
        batch,
        correct.float() / len(cifar100_test_loader.dataset),
        correct_fgsm.float() / len(cifar100_test_loader.dataset),
        correct_pgd.float() / len(cifar100_test_loader.dataset),
        correct_cw.float() / len(cifar100_test_loader.dataset),
        correct_mifgsm.float() / len(cifar100_test_loader.dataset),
        correct_tifgsm.float() / len(cifar100_test_loader.dataset),
        correct_difgsm.float() / len(cifar100_test_loader.dataset),
        correct_nifgsm.float() / len(cifar100_test_loader.dataset),
        eval_finish - eval_start
    ))
    
    return None

def eval_training_avg(batch = 0):

    eval_start = time.time()
    net.eval()
    net.model.eval()
    correct = 0.0
    correct_fgsm = 0.0
    correct_pgd = 0.0
    correct_cw = 0.0
    correct_mifgsm = 0.0
    correct_tifgsm = 0.0
    correct_difgsm = 0.0
    correct_nifgsm = 0.0
    
    
    correct_fgsm_avg = 0.0
    correct_pgd_avg = 0.0

    for (images, labels) in cifar100_test_loader:

        images = images.cuda()
        labels = labels.cuda()
        
        img_fgsm = atk_fgsm_test_avg(images, labels).cuda()
        
        img_pgd = atk_pgd_test_avg(images, labels).cuda()
        
        img_cw = atk_cw_test_avg(images, labels).cuda()
        
        img_mifgsm = atk_mifgsm_test_avg(images, labels).cuda()
        
        img_tifgsm = atk_tifgsm_test_avg(images, labels).cuda()
    
        img_difgsm = atk_difgsm_test_avg(images, labels).cuda()
        
        img_nifgsm = atk_nifgsm_test_avg(images, labels).cuda()
        
        
        
        img_fgsm_avg = atk_fgsm_avg(images, labels).cuda()
        
        img_pgd_avg = atk_pgd_avg(images, labels).cuda()
        
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
        
        
        
        outputs = net(img_fgsm_avg).cuda()
        _, preds = outputs.max(1)
        correct_fgsm_avg += preds.eq(labels).sum()
        
        outputs = net(img_pgd_avg).cuda()
        _, preds = outputs.max(1)
        correct_pgd_avg += preds.eq(labels).sum()
        
    eval_finish = time.time()
    logger.info('batch : {}, Acc: {:.4f}, fgsm : {:.4f}, pgd : {:.4f}, cw : {:.4f}, mifgsm : {:.4f}, tifgsm : {:.4f}, difgsm : {:.4f}, nifgsm : {:.4f}, fgsm_avg : {:.4f}, pgd_avg : {:.4f}, Time consumed:{:.2f}s - avg surrogate model '.format(
        batch,
        correct.float() / len(cifar100_test_loader.dataset),
        correct_fgsm.float() / len(cifar100_test_loader.dataset),
        correct_pgd.float() / len(cifar100_test_loader.dataset),
        correct_cw.float() / len(cifar100_test_loader.dataset),
        correct_mifgsm.float() / len(cifar100_test_loader.dataset),
        correct_tifgsm.float() / len(cifar100_test_loader.dataset),
        correct_difgsm.float() / len(cifar100_test_loader.dataset),
        correct_nifgsm.float() / len(cifar100_test_loader.dataset),
        correct_fgsm_avg.float() / len(cifar100_test_loader.dataset),
        correct_pgd_avg.float() / len(cifar100_test_loader.dataset),
        eval_finish - eval_start
    ))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-lmbd', type=float, default=0.0, help='distortion loss hyper parameter')
    parser.add_argument('-gnorm', type=float, default=1.0, help='convolution matrix normalizing constant. W normalized as : arg.gnrom/sqrt(c)')
    parser.add_argument('-r', type=eval, default=True, choices=[True, False], help = 'set True to repeatedly breached scenario')
    parser.add_argument('-conv',type=eval, default=False, choices = [True, False], help = 'If set True,  convolutional layer will be used instead of ODE module')
    parser.add_argument('-avg', type=str, default ='instance', choices = ['instance','avg','avg_grad'], help = 'asdf')
    args = parser.parse_args()
    
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
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_svhn/checkpoint/wideresnet/SVHN_Thursday_14_September_2023_09h_32m_17s/wideresnet-200-regular.pth'))
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_svhn/checkpoint/inceptionv4/SVHN_Thursday_14_September_2023_09h_31m_51s/inceptionv4-200-regular.pth'))
        
    net = net.eval().cuda()
    for param in net.parameters():
        param.requires_grad = False
    net = net.cuda()

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
        
    if args.conv == True : 
        net = Convmodel(model=net, C=ODE_channels).cuda()
    else : 
        net = Eulermodel(model=net, C=ODE_channels).cuda()
    net_fixed = net.model.eval()
    
    atk_train = pgd(model = net_fixed.eval())
    atk_train.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
    
    atk_fgsm_test = fgsm(model = net_fixed.eval())
    atk_fgsm_test.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
    
    atk_pgd_test = pgd(model = net_fixed.eval(), steps=20)
    atk_pgd_test.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
    
    atk_cw_test = CW(model = net_fixed.eval())
    atk_cw_test.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
    
    atk_mifgsm_test = MIFGSM(model = net_fixed.eval())
    atk_mifgsm_test.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
    
    atk_tifgsm_test = TIFGSM(model = net_fixed.eval())
    atk_tifgsm_test.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 

    atk_difgsm_test = DIFGSM(model = net_fixed.eval())
    atk_difgsm_test.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
    
    atk_nifgsm_test = NIFGSM(model = net_fixed.eval())
    atk_nifgsm_test.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.conv == True :
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/batch_CONV', settings.TIME_NOW)
    else :  
        if args.avg == 'instance' : 
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/batch_ODE', settings.TIME_NOW)
        if args.avg == 'avg' : 
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/batch_ODE_avg', settings.TIME_NOW)
        if args.avg == 'avg_grad' : 
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/batch_ODE_avg_grad', settings.TIME_NOW)
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    logger = get_logger(logpath=os.path.join(checkpoint_path, 'logs'), filepath=os.path.abspath(__file__))
    logger.propagate = False
    logger.info(args)  
        
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    fixed_model_list = []

    for epoch in range(1, 10 + 1):
        fixed_model_list.append(net_fixed)
    
        if epoch > args.warm:
            train_scheduler.step(epoch)
            
        train(epoch)
        if args.conv == True : 
            net_fixed = Convmodel(model=net.model.eval(), C=ODE_channels).cuda()
            net_fixed.copy_conv_model(net)
        else : 
            net_fixed = Eulermodel(model=net.model.eval(), C=ODE_channels).cuda() 
            net_fixed.copy_euler_model(net)
            if epoch == 1 : 
                net_avg = Eulermodel(model=net.model.eval(), C=ODE_channels).cuda() 
                net_avg.copy_euler_model(net)
            else : 
                net_avg.model = net.model.eval()
                net_avg.average_euler_model(net)
                
        for param in net_fixed.parameters():
            param.requires_grad = False
            
        net_fixed = net_fixed.cuda()
        
        ###############
        if args.conv == True : 
            net = Convmodel(model=net.model, C=ODE_channels).cuda()
        else : 
            net = Eulermodel(model=net.model, C=ODE_channels).cuda()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        ################
        
        if args.avg == 'instance' : 
            atk_train = pgd(model = net_fixed.eval())
            atk_train.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        if args.avg == 'avg' :
            atk_train = pgd(model = net_avg.eval())
            atk_train.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        if args.avg == 'avg_grad' :
            atk_train = torchattacks.PGD_avg(fixed_model_list)
            atk_train.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        atk_fgsm_test = fgsm(model = net_fixed.eval())
        atk_fgsm_test.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        atk_pgd_test = pgd(model = net_fixed.eval(), steps=20)
        atk_pgd_test.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        atk_cw_test = CW(model = net_fixed.eval())
        atk_cw_test.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        atk_mifgsm_test = MIFGSM(model = net_fixed.eval())
        atk_mifgsm_test.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        atk_tifgsm_test = TIFGSM(model = net_fixed.eval())
        atk_tifgsm_test.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        atk_difgsm_test = DIFGSM(model = net_fixed.eval())
        atk_difgsm_test.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        atk_nifgsm_test = NIFGSM(model = net_fixed.eval())
        atk_nifgsm_test.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        
        atk_fgsm_test_avg = fgsm(model = net_avg.eval())
        atk_fgsm_test_avg.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        atk_pgd_test_avg = pgd(model = net_avg.eval(), steps=20)
        atk_pgd_test_avg.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        atk_cw_test_avg = CW(model = net_avg.eval())
        atk_cw_test_avg.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        atk_mifgsm_test_avg = MIFGSM(model = net_avg.eval())
        atk_mifgsm_test_avg.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        atk_tifgsm_test_avg = TIFGSM(model = net_avg.eval())
        atk_tifgsm_test_avg.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        atk_difgsm_test_avg = DIFGSM(model = net_avg.eval())
        atk_difgsm_test_avg.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        atk_nifgsm_test_avg = NIFGSM(model = net_avg.eval())
        atk_nifgsm_test_avg.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        
        
        atk_fgsm_avg = torchattacks.FGSM_avg(fixed_model_list)
        atk_fgsm_avg.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
        
        atk_pgd_avg = torchattacks.PGD_avg(fixed_model_list)
        atk_pgd_avg.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 