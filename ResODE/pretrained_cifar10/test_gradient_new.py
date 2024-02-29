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

class Hop_ODEfun_CN_detach(nn.Module): # naive implementation that calculates g at every time forward is called

    def __init__(self, dim=3, eps=0.1):
        super(Hop_ODEfun_CN_detach, self).__init__()
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
        out_0 = out_0.clone().detach()
        #out_0 = self.fx0
        
        out = self.conv(x)
        out = self.relu(out)
        out = F.conv_transpose2d(-out, self.conv.weight, stride=1,padding=1,output_padding=0,bias=None)

        
        out_e = -1*(0.12) * x
        
        return out + out_e + out_0

    def set_x0(self, x0):
        self.x0 = x0
        
    def set_fx0(self, x0):
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


class ODEmodel_detach(nn.Module):
    def __init__(self, model, C):
        super(ODEmodel_detach,self).__init__()
        self.model = model
        self.relu = nn.ReLU()
        
        self.C = C
        
        self.g1 = nn.Parameter(torch.ones(C)).cuda()
        self.ode1 = ODEBlock_t(Hop_ODEfun_CN_detach(dim=C)).cuda()
        
        self.g2 = nn.Parameter(torch.ones(C)).cuda()
        self.ode2 = ODEBlock_t(Hop_ODEfun_CN_detach(dim=C)).cuda()
        
    
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

    def copy_ode_model(self, ode_net):
        self.g1 = ode_net.g1
        self.g2 = ode_net.g2
        self.ode1.odefunc.conv.weight  = ode_net.ode1.odefunc.conv.weight  
        self.ode1.odefunc.conv.weight_g = ode_net.ode1.odefunc.conv.weight_g
        self.ode1.odefunc.conv.weight_v = ode_net.ode1.odefunc.conv.weight_v
        self.ode1.odefunc.conv0.weight_g = ode_net.ode1.odefunc.conv0.weight_g
        self.ode1.odefunc.conv0.weight_v = ode_net.ode1.odefunc.conv0.weight_v
        self.ode1.odefunc.conv0.weight = ode_net.ode1.odefunc.conv0.weight
        
        self.ode2.odefunc.conv.weight  = ode_net.ode2.odefunc.conv.weight
        self.ode2.odefunc.conv.weight_g = ode_net.ode2.odefunc.conv.weight_g
        self.ode2.odefunc.conv.weight_v = ode_net.ode2.odefunc.conv.weight_v
        self.ode2.odefunc.conv.weight_g = ode_net.ode2.odefunc.conv0.weight_g 
        self.ode2.odefunc.conv.weight_v = ode_net.ode2.odefunc.conv0.weight_v
        self.ode2.odefunc.conv.weight = ode_net.ode2.odefunc.conv0.weight

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
        
        
    def copy_ode_model(self, ode_net):
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

        
def train(epoch):

    start = time.time()
    ode_net.train()
    ode_net_detach.train()
    euler_net.train()
    
    adv_error_ode_vs_detach = 0
    adv_error_ode_vs_euler = 0 
    adv_error_euler_vs_detach = 0
    
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            
            ode_net_detach.copy_ode_model(ode_net) 
            euler_net.copy_ode_model(ode_net)
            
            max_val = torch.max(torch.max(torch.max(images))) 
            min_val = torch.min(torch.min(torch.min(images)))
            atk_net.eps = (32/255)/(max_val - min_val)
            adv_img = atk_net((images - min_val)/(max_val - min_val), labels).cuda()
            adv_img = adv_img*(max_val - min_val) + min_val
            
            atk_ode.eps = (32/255)/(max_val - min_val)
            adv_ode = atk_ode((images - min_val)/(max_val - min_val), labels).cuda()
            adv_ode = adv_ode*(max_val - min_val) + min_val
            
            atk_ode_detach.eps = (32/255)/(max_val - min_val)
            adv_detach = atk_ode_detach((images - min_val)/(max_val - min_val), labels).cuda()
            adv_detach = adv_detach*(max_val - min_val) + min_val
            
            atk_euler.eps = (32/255)/(max_val - min_val)
            adv_euler = atk_euler((images - min_val)/(max_val - min_val), labels).cuda()
            adv_euler = adv_euler*(max_val - min_val) + min_val
            
            adv_error_ode_vs_detach += torch.sum( torch.sum( torch.sum( torch.abs( adv_ode - adv_detach ))))
            adv_error_ode_vs_euler += torch.sum( torch.sum( torch.sum( torch.abs( adv_ode - adv_euler ))))
            adv_error_euler_vs_detach += torch.sum( torch.sum( torch.sum( torch.abs( adv_euler - adv_detach ))))
    
        optimizer.zero_grad()
        if epoch > 5 :
            net_optimizer.zero_grad()
        outputs = ode_net(adv_img)
        loss_t = loss_function(outputs, labels)
        loss = loss_t
        
        loss.backward()
        optimizer.step()
        if epoch > 5 : 
            net_optimizer.step()
        if epoch <= args.warm:
            warmup_scheduler.step()
            
    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    print('error ode vs detach : ', adv_error_ode_vs_detach/ len(cifar100_training_loader.dataset), 
    'error ode vs euler : ', adv_error_ode_vs_euler/ len(cifar100_training_loader.dataset), 
    'error detach vs euler : ', adv_error_euler_vs_detach/ len(cifar100_training_loader.dataset)
    )
    
#@torch.no_grad() #grad is needed for calculating adversarial examples
def eval_training(epoch=0, tb=True, before=False):

    start = time.time()
    
    test_loss = 0.0 # cost function error
    correct = 0.0
    correct_ode  = 0.0
    
    for (images, labels) in cifar100_test_loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
            
            max_val = torch.max(torch.max(torch.max(images))) 
            min_val = torch.min(torch.min(torch.min(images)))
            atk_net.eps = (16/255)/(max_val - min_val)
            adv_img = atk_net((images - min_val)/(max_val - min_val), labels).cuda()
            adv_img = adv_img*(max_val - min_val) + min_val
            
        outputs = ode_net(images).cuda()
        
        outputs = net(images).cuda()
        loss = loss_function(outputs, labels)
        
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        
        adv_outputs_ode = ode_net(adv_img).cuda()
        _, preds_ode = adv_outputs_ode.max(1)
        correct_ode += preds_ode.eq(labels).sum()
        
        
    finish = time.time()
    
    print('evaluating nework after training.. transfer samples generated using fixed euler...')
    print('Test set: Epoch: {}, Clean acc : {:.4f}, transfer to ode acc : {:.4f},Time consumed:{:.2f}s'.format(
        epoch,
        correct.float() / len(cifar100_test_loader.dataset),
        correct_ode.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    
    print()
    return correct.float() / len(cifar100_test_loader.dataset)

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
        self.steps = 4
    def forward(self, x):
        x = self.model.forward_1(x)
        
        
        fx01 = self.relu(self.conv0_1(x))
        #fx01 = 0
        for idx in range(self.steps):
            x = x +(1/self.steps)*(-F.conv_transpose2d(self.relu(self.conv_1(x)),  self.conv_1.weight, stride=1,padding=1,output_padding=0,bias=None) + fx01 + -0.12*x )
        x = x*self.g1[None,:,None,None]
        
        fx02 = self.relu(self.conv0_2(x))
        #fx02 = 0
        for idx in range(self.steps):
            x = x +(1/self.steps)*(-F.conv_transpose2d(self.relu(self.conv_2(x)),  self.conv_2.weight, stride=1,padding=1,output_padding=0,bias=None) + fx02 + -0.12*x )
        x = x*self.g2[None,:,None,None]
        
        x = self.relu(x)
        
        
        x = self.model.forward_NN(x)
        return x
        
    def copy_ode_model(self, ode_net):
        self.g1 = torch.nn.Parameter(ode_net.g1.detach().clone())
        self.g2 = torch.nn.Parameter(ode_net.g2.detach().clone())
         
        self.conv_1.weight_g = torch.nn.Parameter(ode_net.ode1.odefunc.conv.weight_g)
        self.conv_1.weight_v = torch.nn.Parameter(ode_net.ode1.odefunc.conv.weight_v)
        self.conv_1.weight  = ode_net.ode1.odefunc.conv.weight 
        self.conv0_1.weight_g = torch.nn.Parameter(ode_net.ode1.odefunc.conv0.weight_g)
        self.conv0_1.weight_v = torch.nn.Parameter(ode_net.ode1.odefunc.conv0.weight_v)
        self.conv0_1.weight = ode_net.ode1.odefunc.conv0.weight
        
        self.conv_2.weight_g = torch.nn.Parameter(ode_net.ode2.odefunc.conv.weight_g)
        self.conv_2.weight_v = torch.nn.Parameter(ode_net.ode2.odefunc.conv.weight_v)
        self.conv_2.weight  = ode_net.ode2.odefunc.conv.weight
        self.conv0_2.weight_g = torch.nn.Parameter(ode_net.ode2.odefunc.conv0.weight_g )
        self.conv0_2.weight_v = torch.nn.Parameter(ode_net.ode2.odefunc.conv0.weight_v)
        self.conv0_2.weight = ode_net.ode2.odefunc.conv0.weight


def get_fgsm_samples(model, images, labels, eps ):

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).cuda() # cifar 10
    std =  torch.tensor([0.2471, 0.2435, 0.2616]).cuda() # cifar 10

    images = ((images * std[None,:,None,None]) + mean[None,:,None,None]) # un-normalize

    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    loss = nn.CrossEntropyLoss()
    
    

    #model = model.clone().detach()

    images.requires_grad = True
    outputs1 = model(images)

    cost = loss(outputs1, labels)

    # Update adversarial images
    grad = torch.autograd.grad( cost, images, retain_graph=False, create_graph=False )[0]

    adv_images = images + eps * grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    
    adv_images = (adv_images - mean[None,:,None,None]) / std[None,:,None,None]
        
    return adv_images  


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

    
    ode_net = ODEmodel(model=net, C=ODE_channels).cuda()
    ode_net_detach = ODEmodel_detach(model = net, C=ODE_channels).cuda()
    ode_net_detach.copy_ode_model(ode_net)
    euler_net = Eulermodel(model=net, C=ODE_channels).cuda() 
    euler_net.copy_ode_model(ode_net)
    
    atk_net = fgsm(model = net)
    atk_net.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    
    atk_euler = fgsm(model = euler_net)
    atk_euler.eps = 16.0/255.0
    
    atk_ode = fgsm(model = ode_net)
    atk_ode.eps = 16.0/255.0
    
    atk_ode_detach = fgsm(model = ode_net_detach)
    atk_ode.eps = 16.0/255.0
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ode_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    
    # train & eval
    for epoch in range(1, settings.EPOCH + 1):
     
        atk_net = fgsm(model = net)
        atk_net.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
        
        if epoch > args.warm:
            train_scheduler.step(epoch)
            
        train(epoch)
        
        
        acc = eval_training(epoch)
        if (epoch > 3 and epoch % 5 == 0):
            '''
            atk_net = fgsm(model = ode_net)
            atk_net.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
            '''
            euler_net = Eulermodel(model=net, C=ODE_channels).cuda() 
            euler_net.copy_ode_model(ode_net)
            atk_net = fgsm(model = euler_net)
            atk_net.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
            
            net_inside = ode_net.model
            for param in net_inside.parameters():
                param.requires_grad = True
                net_optimizer = optim.SGD(net_inside.parameters(), lr=-0.9, momentum=0.9, weight_decay=5e-4) # for fine-tuning net
            
            
    



