#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader

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
            x = x +(self.t/self.steps)*(-F.conv_transpose2d(self.relu(self.conv_2(x)),  self.conv_2.weight, stride=1,padding=1,output_padding=0,bias=None) + self.relu(self.conv0_2(fx02)) + -0.12*x )
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
            x = x +(self.t/self.steps)*(-F.conv_transpose2d(self.relu(self.conv_2(x)),  self.conv_2.weight, stride=1,padding=1,output_padding=0,bias=None) + self.relu(self.conv0_2(fx02)) + -0.12*x )
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()
    
    from models.inceptionv3 import inceptionv3
    net = inceptionv3()
    
    net = Eulermodel(model = net, C=32)
    
    net.load_state_dict(torch.load('./checkpoint/inceptionv3/rODE_loss_t/cifar10_Friday_03_November_2023_04h_33m_56s/inceptionv3-21-regular.pth'))

    print('net_g1 : ',net.g1[0:3])
    print('net_g2 : ',net.g2[0:3])
    print('net_conv0_1_weight_g : ',net.conv0_1.weight_g[0:3])
    print('net_conv0_2_weight_g : ',net.conv0_2.weight_g[0:3])
    
    print('g1 requires grad ', net.g1.requires_grad)