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
from torchdiffeq import odeint_adjoint as odeint


from torchattacks import FGSM as fgsm
from torchattacks import PGD as pgd


class Hop_ODEfun_CN(nn.Module): # naive implementation that calculates g at every time forward is called

    def __init__(self, dim=3, eps=0.1):
        super(Hop_ODEfun_CN, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv  = weight_norm(nn.Conv2d(dim, dim, 3, 1, 1,bias=False))
        #self.conv  = (nn.Conv2d(dim, dim, 3, 1, 1,bias=False))
        self.conv.weight_g = nn.Parameter(torch.ones(dim,1,1,1)/math.sqrt(dim))
        self.conv.weight_g.requires_grad=False
        
        self.conv0 = weight_norm(nn.Conv2d(dim, dim, 1, 1, 0,bias=False)) # k s p
        

        self.nfe = 0
        self.x0 = 0
        self.fx0 = 0
        self.dim = dim
        
        
        self.eps = 0.1

    def forward(self, t, x):
        out_0 = self.conv0(self.x0)
        out_0 = self.relu(out_0)
        #out_0 = self.fx0
        out = self.conv(x)
        out = self.relu(out)
        out = F.conv_transpose2d(-out, self.conv.weight, stride=1,padding=1,output_padding=0,bias=None)

        
        out_e = -1*(self.eps + 2e-2) * x
        
        return out + out_e + out_0

    def set_x0(self, x0):
        self.x0 = x0
    def set_fx0(self, x0) : 
        self.fx0 = self.relu(self.conv0(x0))


class ODEBlock_t(nn.Module):

    def __init__(self, odefunc, tau=1.0):
        super(ODEBlock_t, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, tau]).float()
        self.tau = tau
        self.step_size = self.tau/4.0

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        # time_step = time_step_function(self.epoch)
        self.odefunc.set_x0(x)
        self.odefunc.set_fx0(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol, method='euler')#, options=dict(step_size = self.step_size)) #,  options=dict(step_size=5.0) or options=dict(grid_constructor= prtorch 1dim tensor)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value      
        
class ODEmodel(nn.Module):
    def __init__(self, model, C):
        super(ODEmodel,self).__init__()
        self.model = model
        self.relu = nn.ReLU()
        
        self.C = C
        
        self.g1 = nn.Parameter(torch.ones(C)).cuda()
        self.ode1 = ODEBlock_t(Hop_ODEfun_CN(dim=C)).cuda()
        
        self.g2 = nn.Parameter(torch.ones(C)).cuda()
        self.ode2 = ODEBlock_t(Hop_ODEfun_CN(dim=C)).cuda()
        
    
    def forward(self, x):
        x = self.model.forward_1(x)
        
        x = self.ode1(x)*self.g1[None,:,None,None]
        x = self.ode2(x)*self.g2[None,:,None,None]
        x = self.relu(x)
        
        x = self.model.forward_NN(x)
        return x
    
    
    def forward_L1(self, x):
        x = self.model.forward_1(x)
        
        x_before_ode = x
        
        x = self.ode1(x)*self.g1[None,:,None,None]
        x = self.ode2(x)*self.g2[None,:,None,None]
        x = self.relu(x)
        
        x_after_ode = x
        
        x = self.model.forward_NN(x)
        return x , x_before_ode, x_after_ode
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
    '''
        self.g1 = ode_net.g1
        self.g2 = ode_net.g2
        self.conv_1.weight  = ode_net.ode1.odefunc.conv.weight  
        self.conv_1.weight_g = ode_net.ode1.odefunc.conv.weight_g
        self.conv_1.weight_v = ode_net.ode1.odefunc.conv.weight_v
        self.conv0_1.weight_g = ode_net.ode1.odefunc.conv0.weight_g
        self.conv0_1.weight_v = ode_net.ode1.odefunc.conv0.weight_v
        self.conv0_1.weight = ode_net.ode1.odefunc.conv0.weight
        
        self.conv_2.weight  = ode_net.ode2.odefunc.conv.weight
        self.conv_2.weight_g = ode_net.ode2.odefunc.conv.weight_g
        self.conv_2.weight_v = ode_net.ode2.odefunc.conv.weight_v
        self.conv0_2.weight_g = ode_net.ode2.odefunc.conv0.weight_g 
        self.conv0_2.weight_v = ode_net.ode2.odefunc.conv0.weight_v
        self.conv0_2.weight = ode_net.ode2.odefunc.conv0.weight
    '''

        
def train(epoch):

    start = time.time()
    euler_net.train()
    
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            B,_,_,_ = images.size()
            '''
            max_val = torch.max(torch.max(torch.max(images))) 
            min_val = torch.min(torch.min(torch.min(images)))
            atk_net.eps = (16/255)/(max_val - min_val)
            adv_img = atk_net((images - min_val)/(max_val - min_val), labels).cuda()
            adv_img = adv_img*(max_val - min_val) + min_val
            '''
            adv_img = atk_net(images, labels).cuda()
            
            
            
        optimizer.zero_grad()
        
        adv_img = torch.cat( (images,adv_img), dim=0 )
        outputs, f_before, f_after = euler_net.forward_L1(adv_img)
        _,C,h,w = f_before.size()
        loss_dist = torch.sum(torch.sum(torch.sum(torch.abs(f_before[0:B,:,:,:] - f_after[0:B,:,:,:]))))        
        loss_dist += torch.sum(torch.sum(torch.sum(torch.abs(f_before[0:B,:,:,:] - f_after[0:B,:,:,:]))))   
        loss_t = loss_function(outputs[B:2*B,:], labels)
        loss = loss_t + 0.01*loss_dist/(B*h*w)
        '''
        outputs = euler_net(adv_img)
        loss_t = loss_function(outputs, labels)
        loss = loss_t
        '''
        loss.backward()
        optimizer.step()
        
        
        
        if epoch <= args.warm:
            warmup_scheduler.step()
            
    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    
#@torch.no_grad() #grad is needed for calculating adversarial examples
def eval_training(epoch=0, tb=True, before=False):

    start = time.time()
    euler_net.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0
    correct_ode  = 0.0
    
    for (images, labels) in cifar100_test_loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
            '''
            max_val = torch.max(torch.max(torch.max(images))) 
            min_val = torch.min(torch.min(torch.min(images)))
            atk_net.eps = (16/255)/(max_val - min_val)
            adv_img = atk_net((images - min_val)/(max_val - min_val), labels).cuda()
            adv_img = adv_img*(max_val - min_val) + min_val
            '''
            adv_img = atk_net(images, labels).cuda()
        outputs = euler_net(images).cuda()
        
        outputs = net(images).cuda()
        loss = loss_function(outputs, labels)
        
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        
        adv_outputs_ode = euler_net(adv_img).cuda()
        _, preds_ode = adv_outputs_ode.max(1)
        correct_ode += preds_ode.eq(labels).sum()
        
        
    finish = time.time()
    if before == False : 
        print('evaluating nework after training.. transfer samples generated using fixed euler...')
    else :
        print('evaluating right after breaching.. ')
    print('Test set: Epoch: {}, Clean acc : {:.4f}, transfer to ode acc : {:.4f},Time consumed:{:.2f}s'.format(
        epoch,
        correct.float() / len(cifar100_test_loader.dataset),
        correct_ode.float() / len(cifar100_test_loader.dataset),
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
    parser.add_argument('-lmbd', type=float, default=0.0, help='distortion loss hyper parameter')
    parser.add_argument('-tol', type=float, default=1e-3, help = 'default ODE solver tolerance')
    args = parser.parse_args()
    
    cifar100_training_loader = get_training_dataloader( 
        settings.CIFAR10_TRAIN_MEAN, 
        settings.CIFAR10_TRAIN_STD,
        num_workers=4,
        batch_size=128,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader( 
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=4,
        batch_size=128,
        shuffle=True
    )
    
    from models.resnet import resnet18
    net = resnet18()
    net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_cifar10/checkpoint/resnet18/cifar10_Thursday_14_September_2023_15h_12m_19s/resnet18-200-regular.pth'))

    import copy

    net = net.eval().cuda()
    for param in net.parameters():
        param.requires_grad = False
    net = net.cuda()


    ODE_channels = 64

    euler_net = Eulermodel(model=net, C=ODE_channels).cuda()  
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(euler_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    
    # train & eval
    for epoch in range(1, settings.EPOCH + 1):
     
        atk_net = pgd(model = net.eval())
        atk_net.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
        
        if epoch > args.warm:
            train_scheduler.step(epoch)
            
        train(epoch)
        
        acc = eval_training(epoch)
        if (epoch > 3 and epoch % 5 == 0):
            fixed_net = Eulermodel(model=net, C=ODE_channels).cuda()  
            fixed_net.copy_euler_model(euler_net)
            atk_net = pgd(model = fixed_net.eval())
            atk_net.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
            
            acc = eval_training(epoch, before = True)
            
    