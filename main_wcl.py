from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import models.preact_resnet as models

import os
import sys
import time
import argparse
import datetime

from torch.autograd import Variable

import dataloader

from collections import OrderedDict
import math
import random

parser = argparse.ArgumentParser(description='PyTorch Clothing-1M Training')
parser.add_argument('--lr', default=0.2, type=float, help='learning_rate')
parser.add_argument('--fast_lr', default=0.2, type=float, help='meta learning_rate')
parser.add_argument('--num_fast', default=10, type=int, help='number of random perturbations') # "... the experiments in this paper are conducted using M = 10, as a trade-off between the training speed and the model’s performance"
parser.add_argument('--perturb_ratio', default=0.5, type=float, help='ratio of random perturbations')
parser.add_argument('--num_neighbor', default=10, type=int, help='number of neighbors considered for label transfer')
parser.add_argument('--rampup_epoch', default=20, type=int) # "... ramp up eta (meta-learning rate) from 0 to 0.4 during the first 20 epochs"
parser.add_argument('--lrdecay_epoch', default=80, type=int) # "For each training iteration, we divide the learning rate by 10 after 80 epochs, and train until 120 epochs"
parser.add_argument('--num_epochs', default=120, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--meta_lr', default=0.4, type=float)
parser.add_argument('--gamma_init', default=0.99, type=float, help='Initial exponential moving average weight for the teacher model')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--id', default='wcl')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--noise_ratio', default=0.5, type=float, help='noise ratio')
parser.add_argument('--noise_mode', default='sym', help='sym or asym')
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
parser.add_argument('--checkpoint', default='cross_entropy')
parser.add_argument('--run', default=0, type=int, help='run id (0, 1, 2, 3, or 4) to specify the version of noisy labels')
parser.add_argument('--T', default=10, type=int)
args = parser.parse_args()

random.seed(args.seed)
torch.cuda.set_device(args.gpuid)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_cuda = torch.cuda.is_available()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# Training
def train(epoch):
    global init
    net.train()
    tch_net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    learning_rate = args.lr
    if epoch > args.lrdecay_epoch:
        learning_rate=learning_rate/10
        
    if epoch>args.rampup_epoch:
        meta_lr = args.meta_lr
        gamma = 0.999
    else:
        u = epoch/args.rampup_epoch
        meta_lr = args.meta_lr*math.exp(-5*(1-u)**2)
        gamma = args.gamma_init

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print('\n=> %s Training Epoch #%d, LR=%.6f' %(args.id,epoch, learning_rate))
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() 
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        
        class_loss = criterion(outputs, targets)  # Loss
        class_loss.backward(retain_graph=True)
        
        if epoch > 2:
            if init:
                init = False
                for param,param_tch in zip(net.parameters(),tch_net.parameters()):
                    param_tch.data.copy_(param.data)
            else:
                for param,param_tch in zip(net.parameters(),tch_net.parameters()):
                    param_tch.data.mul_(gamma).add_((1-gamma), param.data)
            
            _,feats = pretrain_net(inputs,get_feat=True)
            tch_outputs = tch_net(inputs,get_feat=False)
            p_tch = F.softmax(tch_outputs,dim=1)
            p_tch = p_tch.detach()
            
            for i in range(args.num_fast):
                # print('---------- i=%d ----------' % i)
                targets_fast = targets.clone()
                randidx = torch.randperm(targets.size(0))
                # print('targets.size:', targets.size())
                loss_weights = torch.cuda.FloatTensor(targets.size()).fill_(0.5)
                # print('loss_weights.requires_grad: ', loss_weights.requires_grad)
                # print('loss_weights:', loss_weights)
                for n in range(int(targets.size(0)*args.perturb_ratio)):
                    # num_neighbor = 10
                    idx = randidx[n]
                    # print('    ------ n=%d (idx: %d)------' % (n, idx))
                    # print('targets[%d]:' % idx, targets[idx])
                    feat = feats[idx]
                    # print('feat.shape:', feat.shape)
                    feat.view(1,feat.size(0))
                    # print('feat.shape:', feat.shape)
                    feat.data = feat.data.expand(targets.size(0),feat.size(0))
                    dist = torch.sum((feat-feats)**2,dim=1)
                    # print('dist.shape:', dist.shape)
                    _, neighbor = torch.topk(dist.data,args.num_neighbor+1,largest=False)
                    # print('neighbor:', neighbor)
                    targets_fast[idx] = targets[neighbor[random.randint(1,args.num_neighbor)]]
                    # print('targets_fast[%d]:' % idx, targets_fast[idx])
                    # for neighbor_idx in range(args.num_neighbor+1):
                    #     print('targets[%d]:' % neighbor[neighbor_idx], targets[neighbor[neighbor_idx]])
                    neighbor_labels = torch.gather(targets, 0, neighbor)
                    # print('neighbor_labels:', neighbor_labels)
                    label_histogram = torch.bincount(neighbor_labels)
                    # print('label_histogram:', label_histogram)
                    frac_old = torch.true_divide(label_histogram[targets[idx]], args.num_neighbor+1)
                    # print('frac_old:', frac_old)
                    frac_new = torch.true_divide(label_histogram[targets_fast[idx]], args.num_neighbor+1)
                    # print('frac_new:', frac_new)
                    loss_weights[idx] = torch.sigmoid(torch.log(torch.true_divide(label_histogram[targets[idx]], label_histogram[targets_fast[idx]])) * args.T)
                    # print('loss_weights[%d]:' % idx, loss_weights[idx])
                    
                fast_loss = criterion(outputs,targets_fast)
                
                grads = torch.autograd.grad(fast_loss, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
                # grads = torch.autograd.grad(fast_loss, net.parameters())
                
                # grads_list = list(grads)
                # print('grads_list')
                # print(len(grads_list))
                # for grad in grads_list:
                #     print(grad.shape)
                
                for grad in grads:
                    grad = grad.detach()
                    grad.requires_grad = False
                fast_weights = OrderedDict((name, param - args.fast_lr*grad) for ((name, param), grad) in zip(net.named_parameters(), grads))
                # grads_temp = [grad.detach() for grad in grads]
                # for grad in grads_temp:
                #     grad.requires_grad = False
                # fast_weights = OrderedDict((name, param - args.fast_lr*grad) for ((name, param), grad) in zip(net.named_parameters(), grads_temp))
                
                fast_out = net.forward(inputs,fast_weights)  
                
                logp_fast = F.log_softmax(fast_out,dim=1)
                
                if i == 0:
                    consistent_loss = torch.matmul(torch.mean(consistent_criterion(logp_fast,p_tch), dim=1), loss_weights)
                else:
                    consistent_loss = consistent_loss + torch.matmul(torch.mean(consistent_criterion(logp_fast,p_tch), dim=1), loss_weights)
            
            meta_loss = consistent_loss*meta_lr/args.num_fast 
            
            meta_loss.backward()
            
        optimizer.step() # Optimizer update
        
        # train_loss += class_loss.data[0]
        train_loss += class_loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        # sys.stdout.write('\r')
        # sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
        #         # %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, class_loss.data[0], 100.*correct/total))
        #         %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, class_loss.data.item(), 100.*correct/total))
        # sys.stdout.flush()
        if batch_idx%10==0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, class_loss.data.item(), 100.*correct/total))
        if batch_idx%50==0:
            with torch.no_grad():
                val(epoch,batch_idx)
                val_tch(epoch,batch_idx)
            net.train()
            tch_net.train()
            
            
def val(epoch,iteration):
    global best
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        # val_loss += loss.data[0]
        val_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
    # Save checkpoint when best model
    acc = 100.*correct/total
    # print("\n| Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, iteration, loss.data[0], acc))
    print("\n| Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, iteration, loss.data.item(), acc))
    # record.write('Epoch #%d Batch #%3d  Acc: %.2f' %(epoch,iteration,acc))
    # print('Epoch #%d Batch #%3d  Acc: %.2f' %(epoch,iteration,acc))
    if acc > best:
        best = acc
        print('| Saving Best Model (net)...')
        save_checkpoint({
            'state_dict': net.state_dict(),
            'best_acc': best,
        }, save_point)

def val_tch(epoch,iteration):
    global best
    tch_net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = tch_net(inputs)
        loss = criterion(outputs, targets)
        
        # val_loss += loss.data[0]
        val_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
    # Save checkpoint when best model
    acc = 100.*correct/total
    # print("| tch Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%\n" %(epoch, iteration, loss.data[0], acc))
    print("| tch Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, iteration, loss.data.item(), acc))
    # record.write(' | tchAcc: %.2f\n' %acc)
    # record.flush()
    # print(' | tchAcc: %.2f\n' %acc)
    if acc > best:
        best = acc
        print('| Saving Best Model (tchnet)...')
        save_checkpoint({
            'state_dict': tch_net.state_dict(),
            'best_acc': best,
        }, save_point)

def test():
    test_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = test_net(inputs)
        loss = criterion(outputs, targets)
        
        # test_loss += loss.data[0]
        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    test_acc = 100.*correct/total   
    print('* Test results : Acc@1 = %.2f%%' %(test_acc))
    # record.write('\nTest Acc: %f\n'%test_acc)
    # record.flush()
    # print('\nTest Acc: %f\n'%test_acc)
    
# ===============================================
# record=open('./checkpoint/'+args.id+'.txt','w')
# record.write('learning rate: %f\n'%args.lr)
# record.write('batch size: %f\n'%args.batch_size)
# record.write('start iter: %d\n'%args.start_iter)
# record.write('mid iter: %d\n'%args.mid_iter)
# record.flush()
print('learning rate: %f\n'%args.lr)
print('batch size: %d\n'%args.batch_size)
print('number of additional mini-batches: %d\n'%args.num_fast)
print('perturbation ratio: %f\n'%args.perturb_ratio)
print('number of neighbor: %d\n'%args.num_neighbor)
print('ramp-up end epoch of the meta-learning rate: %d\n'%args.rampup_epoch)
print('LR decay epoch: %d\n'%args.lrdecay_epoch)

save_point = './checkpoint/%s_%s%s_run%d_M%dn%drho%s.pth.tar'%(args.id,
                                                               args.noise_mode,
                                                               ''.join(str(args.noise_ratio).split('.')),
                                                               args.run,
                                                               args.num_fast,
                                                               args.num_neighbor,
                                                               ''.join(str(args.perturb_ratio).split('.')))

loader = dataloader.cifar_dataloader(dataset=args.dataset,
                                     noise_ratio=args.noise_ratio,
                                     noise_mode=args.noise_mode,
                                     batch_size=args.batch_size,
                                     num_workers=0,
                                     root_dir=args.data_path,
                                     # log=stats_log,
                                     train_val_split_file='%s/train_val_split.json'%args.data_path,
                                     noise_file='%s/%s%s_run%d.json'%(args.data_path,
                                                                      args.noise_mode,
                                                                      ''.join(str(args.noise_ratio).split('.')),
                                                                      args.run))
train_loader,val_loader,test_loader = loader.run()

best = 0
init = True
# Model
print('\nModel setup')
print('| Building net')
net = models.PreActResNet32()
tch_net = models.PreActResNet32()
pretrain_net = models.PreActResNet32()
test_net = models.PreActResNet32()

print('| load pretrain from checkpoint...')
checkpoint = torch.load('./checkpoint/%s.pth.tar'%args.checkpoint)
pretrain_net.load_state_dict(checkpoint['state_dict'])

if use_cuda:
    net.cuda()
    tch_net.cuda()
    pretrain_net.cuda()
    test_net.cuda()
    cudnn.benchmark = True
pretrain_net.eval()

for param in tch_net.parameters():
    param.requires_grad = False
for param in pretrain_net.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
consistent_criterion = nn.KLDivLoss(reduction='none')
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

print('\nTraining model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))

for epoch in range(1, 1+args.num_epochs):
    train(epoch)
    if epoch%2==0:
        print('\nTesting model')
        best_model = torch.load(save_point)
        test_net.load_state_dict(best_model['state_dict'])
        with torch.no_grad():
            test()

# record.close()
