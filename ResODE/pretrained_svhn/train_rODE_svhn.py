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
        self.dim = dim
        
        
        self.eps = 0.1
        #self.eps = nn.Parameter(torch.tensor(eps))    # LW
        #self.eps = nn.Parameter(torch.ones(dim)*eps) # CW
        
        #self.g = nn.Parameter(torch.ones(dim)/math.sqrt(dim))

    def forward(self, t, x):
        out_0 = self.conv0(self.x0)
        out_0 = self.relu(out_0)

        out = self.conv(x)
        out = self.relu(out)
        out = F.conv_transpose2d(-out, self.conv.weight, stride=1,padding=1,output_padding=0,bias=None)
        #out = self.g[None,:,None,None] * out 

        
        out_e = -1*(self.eps + 2e-2) * x           # LW
        #out_e = -(self.eps + 1e-3)[None,:,None,None] * x # CW
        
        return out + out_e + out_0

    def set_x0(self, x0):
        self.x0 = x0

class ODEBlock_t(nn.Module):

    def __init__(self, odefunc, tau=1.0):
        super(ODEBlock_t, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, tau]).float()
        self.tau = tau
        #self.step_size = self.tau/2.0

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        # time_step = time_step_function(self.epoch)
        self.odefunc.set_x0(x)
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
        
        
def train(epoch):

    start = time.time()
    
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.lmbd > 0 : 
            labels = labels.cuda()
            images = images.cuda()
            B,_,_,_ = images.size()
            
            max_val = torch.max(torch.max(torch.max(images))) 
            min_val = torch.min(torch.min(torch.min(images)))
            atk_fgsm.eps = (8/255)/(max_val - min_val)
            adv_images = atk_fgsm((images - min_val)/(max_val - min_val), labels).cuda()
            adv_images = adv_images*(max_val - min_val) + min_val
            images = torch.cat( (images,adv_images) ,dim=0)

            optimizer.zero_grad()
            outputs, f_before_ode, f_after_ode = net.forward_L1(images)
        
            _,_,h,w = f_before_ode.size()
        
            loss = loss_function(outputs[0:B,:], labels)
            
            loss_t = loss_function(outputs[B:2*B,:], labels)
            
            f_clean = torch.split(f_before_ode,B,dim=0)[0].cuda()
            f_after_ode_split = torch.split(f_after_ode,B,dim=0)
            f_dist = f_after_ode_split[0].cuda()
            f_denoised = f_after_ode_split[1].cuda()
            
            dist_loss =  torch.sum( torch.abs(f_clean - f_dist), dim=(0,1,2,3))/(B*h*w) + torch.sum( torch.abs(f_clean - f_denoised), dim=(0,1,2,3))/(B*h*w)
            
            loss = loss_t + args.lmbd*dist_loss
        else : 
            labels = labels.cuda()
            images = images.cuda()
            B,_,_,_ = images.size()
            adv_images = atk_fgsm(images, labels).cuda()
        
            optimizer.zero_grad()
            outputs = net(adv_images)
            loss_t = loss_function(outputs, labels)
            loss = loss_t
        
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))


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
        -1.0,#correct_fgsm.float() / len(cifar100_test_loader.dataset),
        -1.0,#correct_pgd.float() / len(cifar100_test_loader.dataset),
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
    parser.add_argument('-lmbd', type=float, default=0.01, help='distortion loss hyper parameter')
    parser.add_argument('-tol', type=float, default=1e-3, help = 'default ODE solver tolerance')
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
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_svhn/checkpoint/inceptionv4/SVHN_Thursday_14_September_2023_09h_31m_51s/inceptionv4-200-regular.pth'))
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
        net.load_state_dict(torch.load('/home/yhshin/ResODE/pretrained_svhn/checkpoint/wideresnet/SVHN_Thursday_14_September_2023_09h_32m_17s/wideresnet-200-regular.pth'))
        
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
        ODE_channels = 192

    if args.net == 'inceptionv4' : 
        ODE_channels = 384 # 192 + 192
        
    if args.net == 'inceptionresnetv2' : 
        ODE_channels = 384 # 192 + 192
        
    if args.net == 'wideresnet' : 
        ODE_channels = 160
        
    net = ODEmodel(model=net, C=ODE_channels).cuda()
    
    atk_fgsm = fgsm(model = net.model)
    atk_fgsm.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
    atk_pgd = pgd(model = net.model)
    atk_pgd.set_normalization_used(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]) # svhn 
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        if args.lmbd > 0 :
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/rODE_dist_loss', settings.TIME_NOW)
        elif args.lmbd == 0 : 
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+'/rODE_loss_t', settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
            
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, 80 + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue
            
        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
    
    writer.close()
    