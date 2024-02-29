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
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

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

        
        out_e = -1*(0.12) * x
        
        return out + out_e + out_0

    def set_x0(self, x0):
        self.x0 = x0
        
    def set_fx0(self, x0):
        self.fx0 = self.relu(self.conv0(x0))
        
class Hop_ODEfun_CN_fx0(nn.Module): # naive implementation that calculates g at every time forward is called

    def __init__(self, dim=3, eps=0.1):
        super(Hop_ODEfun_CN_fx0, self).__init__()
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
        '''
        out_0 = self.conv0(self.x0)
        out_0 = self.relu(out_0)
        '''
        out_0 = self.fx0
        
        out = self.conv(x)
        out = self.relu(out)
        out = F.conv_transpose2d(-out, self.conv.weight, stride=1,padding=1,output_padding=0,bias=None)

        
        out_e = -1*(0.12) * x
        
        return out + out_e + out_0

    def set_x0(self, x0):
        self.x0 = x0
        
    def set_fx0(self, x0):
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
        
        #out_0 = self.conv0(self.x0)
        #out_0 = self.relu(out_0)
        
        out_0 = self.fx0.detach().clone()
        
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
        self.step_size = self.tau/1.0

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        # time_step = time_step_function(self.epoch)
        self.odefunc.set_x0(x)
        self.odefunc.set_fx0(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol, method='euler', options=dict(step_size=0.25) )#, options=dict(step_size=0.1)) #,  options=dict(step_size=5.0) or options=dict(grid_constructor= prtorch 1dim tensor)
        
        #print('difference between linear interpolated output and just output : ', torch.sum(torch.sum(torch.sum(torch.abs(out[0]-out[1]))))/500 )
        
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
    def copy_ode_model(self, ode_net):
        self.g1 = ode_net.g1
        self.g2 = ode_net.g2
         
        self.ode1.odefunc.conv.weight_g = ode_net.ode1.odefunc.conv.weight_g
        self.ode1.odefunc.conv.weight_v = ode_net.ode1.odefunc.conv.weight_v
        self.ode1.odefunc.conv.weight  = ode_net.ode1.odefunc.conv.weight
        self.ode1.odefunc.conv0.weight_g = ode_net.ode1.odefunc.conv0.weight_g
        self.ode1.odefunc.conv0.weight_v = ode_net.ode1.odefunc.conv0.weight_v
        self.ode1.odefunc.conv0.weight = ode_net.ode1.odefunc.conv0.weight
        
        
        self.ode2.odefunc.conv.weight_g.data = ode_net.ode2.odefunc.conv.weight_g.data
        self.ode2.odefunc.conv.weight_v.data = ode_net.ode2.odefunc.conv.weight_v.data
        self.ode2.odefunc.conv.weight.data  = ode_net.ode2.odefunc.conv.weight.data
        self.ode2.odefunc.conv0.weight_g.data  = ode_net.ode2.odefunc.conv0.weight_g.data
        self.ode2.odefunc.conv0.weight_v.data = ode_net.ode2.odefunc.conv0.weight_v.data
        self.ode2.odefunc.conv0.weight.data = ode_net.ode2.odefunc.conv0.weight.data
'''
    def copy_ode_model(self, ode_net):
        self.g1 = nn.Parameter(ode_net.g1.detach().clone())
        self.g2 = nn.Parameter(ode_net.g2.detach().clone())
         
        self.ode1.odefunc.conv.weight_g = nn.Parameter(ode_net.ode1.odefunc.conv.weight_g.detach().clone())
        self.ode1.odefunc.conv.weight_v = nn.Parameter(ode_net.ode1.odefunc.conv.weight_v.detach().clone())
        self.ode1.odefunc.conv.weight  = ode_net.ode1.odefunc.conv.weight 
        self.ode1.odefunc.conv0.weight_g = nn.Parameter(ode_net.ode1.odefunc.conv0.weight_g.detach().clone())
        self.ode1.odefunc.conv0.weight_v = nn.Parameter(ode_net.ode1.odefunc.conv0.weight_v.detach().clone())
        self.ode1.odefunc.conv0.weight = ode_net.ode1.odefunc.conv0.weight
        
        
        self.ode2.odefunc.conv.weight_g = nn.Parameter(ode_net.ode2.odefunc.conv.weight_g.detach().clone())
        self.ode2.odefunc.conv.weight_v = nn.Parameter(ode_net.ode2.odefunc.conv.weight_v.detach().clone())
        self.ode2.odefunc.conv.weight  = ode_net.ode2.odefunc.conv.weight
        self.ode2.odefunc.conv0.weight_g  = nn.Parameter(ode_net.ode2.odefunc.conv0.weight_g.detach().clone())
        self.ode2.odefunc.conv0.weight_v = nn.Parameter(ode_net.ode2.odefunc.conv0.weight_v.detach().clone())
        self.ode2.odefunc.conv0.weight = ode_net.ode2.odefunc.conv0.weight
''' 
class ODEmodel_fx0(nn.Module):
    def __init__(self, model, C):
        super(ODEmodel_fx0,self).__init__()
        self.model = model
        self.relu = nn.ReLU()
        
        self.C = C
        
        self.g1 = nn.Parameter(torch.ones(C)).cuda()
        self.ode1 = ODEBlock_t(Hop_ODEfun_CN_fx0(dim=C)).cuda()
        
        self.g2 = nn.Parameter(torch.ones(C)).cuda()
        self.ode2 = ODEBlock_t(Hop_ODEfun_CN_fx0(dim=C)).cuda()
        
    
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
        self.g1.data = ode_net.g1.data
        self.g2.data = ode_net.g2.data
         
        self.ode1.odefunc.conv.weight_g.data = ode_net.ode1.odefunc.conv.weight_g.data
        self.ode1.odefunc.conv.weight_v.data = ode_net.ode1.odefunc.conv.weight_v.data
        self.ode1.odefunc.conv.weight.data  = ode_net.ode1.odefunc.conv.weight.data
        self.ode1.odefunc.conv0.weight_g.data = ode_net.ode1.odefunc.conv0.weight_g.data
        self.ode1.odefunc.conv0.weight_v.data = ode_net.ode1.odefunc.conv0.weight_v.data
        self.ode1.odefunc.conv0.weight.datat = ode_net.ode1.odefunc.conv0.weight.data
        
        
        self.ode2.odefunc.conv.weight_g.data = ode_net.ode2.odefunc.conv.weight_g.data
        self.ode2.odefunc.conv.weight_v.data = ode_net.ode2.odefunc.conv.weight_v.data
        self.ode2.odefunc.conv.weight.data  = ode_net.ode2.odefunc.conv.weight.data
        self.ode2.odefunc.conv0.weight_g.data  = ode_net.ode2.odefunc.conv0.weight_g.data
        self.ode2.odefunc.conv0.weight_v.data = ode_net.ode2.odefunc.conv0.weight_v.data
        self.ode2.odefunc.conv0.weight.data = ode_net.ode2.odefunc.conv0.weight.data

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
        self.g1.data = ode_net.g1.data
        self.g2.data = ode_net.g2.data
         
        self.ode1.odefunc.conv.weight_g.data = ode_net.ode1.odefunc.conv.weight_g.data
        self.ode1.odefunc.conv.weight_v.data = ode_net.ode1.odefunc.conv.weight_v.data
        self.ode1.odefunc.conv.weight.data  = ode_net.ode1.odefunc.conv.weight.data
        self.ode1.odefunc.conv0.weight_g.data = ode_net.ode1.odefunc.conv0.weight_g.data
        self.ode1.odefunc.conv0.weight_v.data = ode_net.ode1.odefunc.conv0.weight_v.data
        self.ode1.odefunc.conv0.weight.datat = ode_net.ode1.odefunc.conv0.weight.data
        
        
        self.ode2.odefunc.conv.weight_g.data = ode_net.ode2.odefunc.conv.weight_g.data
        self.ode2.odefunc.conv.weight_v.data = ode_net.ode2.odefunc.conv.weight_v.data
        self.ode2.odefunc.conv.weight.data  = ode_net.ode2.odefunc.conv.weight.data
        self.ode2.odefunc.conv0.weight_g.data  = ode_net.ode2.odefunc.conv0.weight_g.data
        self.ode2.odefunc.conv0.weight_v.data = ode_net.ode2.odefunc.conv0.weight_v.data
        self.ode2.odefunc.conv0.weight.data = ode_net.ode2.odefunc.conv0.weight.data

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
        self.steps = 2
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
        self.g1.data =ode_net.g1.data
        self.g2.data = ode_net.g2.data
         
        self.conv_1.weight_g.data = ode_net.ode1.odefunc.conv.weight_g.data
        self.conv_1.weight_v.data = ode_net.ode1.odefunc.conv.weight_v.data
        self.conv_1.weight.data  = ode_net.ode1.odefunc.conv.weight.data
        self.conv0_1.weight_g.data = ode_net.ode1.odefunc.conv0.weight_g.data
        self.conv0_1.weight_v.data = ode_net.ode1.odefunc.conv0.weight_v.data
        self.conv0_1.weight.data = ode_net.ode1.odefunc.conv0.weight
        
        self.conv_2.weight_g.data = ode_net.ode2.odefunc.conv.weight_g.data
        self.conv_2.weight_v.data = ode_net.ode2.odefunc.conv.weight_v.data
        self.conv_2.weight.data  = ode_net.ode2.odefunc.conv.weight.data
        self.conv0_2.weight_g.data = ode_net.ode2.odefunc.conv0.weight_g.data
        self.conv0_2.weight_v.data = ode_net.ode2.odefunc.conv0.weight_v.data
        self.conv0_2.weight.data = ode_net.ode2.odefunc.conv0.weight.data

        
def train(epoch):

    start = time.time()
    ode_net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            
            max_val = torch.max(torch.max(torch.max(images))) 
            min_val = torch.min(torch.min(torch.min(images)))
            atk_net.eps = (16/255)/(max_val - min_val)
            adv_img = atk_net((images - min_val)/(max_val - min_val), labels).cuda()
            adv_img = adv_img*(max_val - min_val) + min_val
    
        optimizer.zero_grad()
        outputs = ode_net(adv_img)
        loss_t = loss_function(outputs, labels)
        loss = loss_t
        
        loss.backward()
        optimizer.step()
        
        if epoch <= args.warm:
            warmup_scheduler.step()
            
    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    
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
    
def get_fgsm_samples(model, images, labels):

    images = random_imgs.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    loss = nn.CrossEntropyLoss()

    #model = model.clone().detach()

    images.requires_grad = True
    outputs1 = model(images)

    cost = loss(outputs1, labels)

    # Update adversarial images
    grad = torch.autograd.grad( cost, images, retain_graph=False, create_graph=False )[0]

    adv_images = images + 8.0/255 * grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
    return adv_images    

if __name__ == '__main__':

    #torch.backends.cudnn.deterministic = True

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
    #opt_ode = optim.SGD(ode_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    ode_net2 = ODEmodel(model=net, C=ODE_channels).cuda()
    ode_net2.copy_ode_model(ode_net)
    #opt_ode2 = optim.SGD(ode_net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    euler_net_nstep  = Eulermodel(model=net, C=ODE_channels).cuda()
    euler_net_nstep.copy_ode_model(ode_net)
    #opt_eulr = optim.SGD(euler_net_nstep.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    euler_net_nstep2 = Eulermodel(model=net, C=ODE_channels).cuda()
    euler_net_nstep2.copy_ode_model(ode_net)
    #opt_eulr2 = optim.SGD(euler_net_nstep2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    ode_net_fx0 = ODEmodel_fx0(model=net, C=ODE_channels).cuda()
    ode_net_fx0.copy_ode_model(ode_net)
    #opt_odefx0 = optim.SGD(ode_net_fx0.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    ode_net_detach = ODEmodel_detach(model=net, C=ODE_channels).cuda()
    ode_net_detach.copy_ode_model(ode_net)
    #opt_ode_detach = optim.SGD(ode_net_detach.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    

    random_imgs = torch.rand(100,3,32,32).cuda() 
    
    net = net.eval().cuda()
    for param in net.parameters():
        param.requires_grad = True
    net = net.cuda()
    
    ode_net.model = net
    euler_net_nstep.model = net
    euler_net_nstep2.model = net
    ode_net_fx0.model = net
    ode_net_detach.model = net
    
    ode_net.model = net
    euler_net_nstep.model = net
    euler_net_nstep2.model = net
    ode_net_fx0.model = net
    ode_net_detach.model = net
    
    atk_ode = fgsm(model = ode_net.eval())
    atk_ode2 = fgsm(model = ode_net2.eval())
    atk_euler = fgsm(model = euler_net_nstep.eval())
    atk_euler2 = fgsm(model = euler_net_nstep2.eval())
    atk_ode_fx0 = fgsm(model = ode_net_fx0.eval())
    atk_ode_detach = fgsm(model = ode_net_detach.eval())
    atk_net = fgsm(model = net.eval())
    '''
    atk_ode = pgd(model = ode_net, random_start=False)
    atk_ode2 = pgd(model = ode_net2, random_start=False)
    atk_euler = pgd(model = euler_net_nstep, random_start=False)
    atk_euler2 = pgd(model = euler_net_nstep2, random_start=False)
    atk_ode_fx0 = pgd(model = ode_net_fx0, random_start=False)
    atk_ode_detach = pgd(model = ode_net_detach, random_start=False)
    atk_net = pgd(model = net,  random_start=False)
    '''
    '''
    atk_ode.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    atk_ode2.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    atk_euler.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    atk_euler2.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    atk_ode_fx0.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    atk_ode_detach.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    atk_net.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    '''
    
    
    '''
    import random
    seed = 34
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    '''
     
    #################################
    ###  only                     ###
    ###    ODE modules            ###
    ###      have bug with        ###
    ###        torchattacks       ###
    ###          API              ###
    #################################
     
    for idx in range(4,5):
        
        labels = torch.tensor(
        [
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
        ]).cuda()
        '''
        net_out1 = net(random_imgs)
        adv_out_net = atk_net(random_imgs, labels)
        adv_out_net2 = atk_net(random_imgs, labels)
        adv_out_net_3 = get_fgsm_samples(net, random_imgs, labels)
        net_out2 = net(random_imgs)
        print('net bug test : ',torch.sum(torch.sum(torch.abs(adv_out_net   - adv_out_net2   )))/500)
        print('net bug test : ',torch.sum(torch.sum(torch.abs(adv_out_net_3   - adv_out_net2   )))/500)
        print('net bug test : ',torch.sum(torch.sum(torch.abs(net_out1   - net_out2   )))/500)
        '''
        
        euler_net_nstep.steps = idx
        euler_net_nstep2.steps = idx
        
        out_ode     = ode_net(random_imgs)
        out_nstep     = euler_net_nstep(random_imgs)
        out_fx0     = ode_net_fx0(random_imgs)
        out_detach     = ode_net_detach(random_imgs)
        out_ode2     = ode_net2(random_imgs)
        out_nstep2     = euler_net_nstep2(random_imgs)
        '''
        adv_ode     = get_fgsm_samples(ode_net, random_imgs, labels)
        adv_out_ode = ode_net(adv_ode)
        adv_nstep       = get_fgsm_samples(euler_net_nstep, random_imgs, labels)
        adv_out_nstep   = euler_net_nstep(adv_nstep)
        adv_fx0       = get_fgsm_samples(ode_net_fx0, random_imgs, labels)
        adv_out_fx0   = ode_net_fx0(adv_fx0)
        adv_detach     = get_fgsm_samples(ode_net_detach, random_imgs, labels)
        adv_out_detach   = ode_net_detach(adv_detach)
        adv_ode2     = get_fgsm_samples(ode_net2, random_imgs, labels)
        adv_out_ode2   = ode_net2(adv_ode2)
        adv_nstep2     = get_fgsm_samples(euler_net_nstep2, random_imgs, labels)
        adv_out_nstep2 = euler_net_nstep2(adv_nstep2)
        '''  
        
        
        adv_nstep     = atk_euler(random_imgs,labels).cuda()
        adv_out_nstep = euler_net_nstep(adv_nstep) 
        adv_fx0     = atk_ode_fx0(random_imgs,labels).cuda()
        adv_out_fx0 = ode_net_fx0(adv_fx0)
        adv_ode_detach = atk_ode_detach(random_imgs,labels).cuda()
        adv_out_detach = ode_net_detach(adv_ode_detach)
        adv_ode2     = atk_ode2(random_imgs,labels)
        adv_out_ode2 = ode_net2(adv_ode2)
        adv_ode     = atk_ode(random_imgs,labels)
        adv_out_ode = ode_net(adv_ode)
        adv_nstep2     = atk_euler2(random_imgs,labels).cuda()
        adv_out_nstep2 = euler_net_nstep2(adv_nstep2)
        
        print('cln logit error ode vs euler  : ', torch.sum(torch.sum(torch.abs(out_ode   - out_nstep  )))/500)
        
        print('cln logit error ode1 vs ode2  : ', torch.sum(torch.sum(torch.abs(out_ode   - out_ode2   )))/500)
        print('cln logit error eul1 vs eul2  : ', torch.sum(torch.sum(torch.abs(out_nstep - out_nstep2 )))/500)
        print('cln logit error ode1 vs eul2  : ', torch.sum(torch.sum(torch.abs(out_ode   - out_nstep2 )))/500)
        print('cln logit error ode2 vs eul1  : ', torch.sum(torch.sum(torch.abs(out_ode2  - out_nstep  )))/500)
        
        print('cln logit error ode2 vs eul2  : ', torch.sum(torch.sum(torch.abs(out_ode2  - out_nstep2  )))/500)
        
        print()
        
        print('adv logit error ode vs euler  : ', torch.sum(torch.sum(torch.abs(adv_out_ode -adv_out_nstep)))/500 )
        
        print('adv logit error ode vs detach : ', torch.sum(torch.sum(torch.abs(adv_out_ode -adv_out_fx0)))/500 )
        print('adv logit error ode vs fx0    : ', torch.sum(torch.sum(torch.abs(adv_out_ode -adv_out_detach)))/500 )
        
        print('adv logit error eul vs detach : ', torch.sum(torch.sum(torch.abs(adv_out_nstep -adv_out_fx0)))/500 )
        print('adv logit error eul vs fx0    : ', torch.sum(torch.sum(torch.abs(adv_out_nstep -adv_out_detach)))/500 )
        
        
        print('adv logit error eul vs eul2   : ', torch.sum(torch.sum(torch.abs(adv_out_nstep -adv_out_nstep2)))/500 )
        print('adv logit error ode vs ode2   : ', torch.sum(torch.sum(torch.abs(adv_out_ode -adv_out_ode2)))/500 )
        print('adv logit error fx0 vs detach : ', torch.sum(torch.sum(torch.abs(adv_out_fx0 -adv_out_detach)))/500 )  
        print('') 
        
    
    
    
    
    
    
    
    
    
    