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

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.04, type=float, help='learning_rate')
parser.add_argument('--start_epoch', default=10, type=int)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--optim_type', default='SGD')
parser.add_argument('--seed', default=7)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--id', default='ce')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--noise_ratio', default=0.5, type=float, help='noise ratio')
parser.add_argument('--noise_mode', default='sym', help='sym or asym')
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
parser.add_argument('--run', default=0, type=int, help='run id (0, 1, 2, 3, or 4) to specify the version of noisy labels')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
  
# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    learning_rate = args.lr
    if epoch > args.start_epoch:
        learning_rate=learning_rate/10        
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('\n=> %s Training Epoch #%d, LR=%.4f' %(args.id,epoch, learning_rate))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        # train_loss += loss.data[0]
        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # sys.stdout.write('\r')
        # sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
        #         %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, loss.data[0], 100.*correct/total))
        # sys.stdout.flush()
        # if batch_idx%1000==0:
        #     val(epoch)
        #     net.train()
        if batch_idx%10==0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, loss.data.item(), 100.*correct/total))
        if batch_idx%50==0:
            with torch.no_grad():
                val(epoch)
            net.train()
            
def val(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # test_loss += loss.data[0]
        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    # print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data[0], acc))
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data.item(), acc))
    # record.write('Validation Acc: %f\n'%acc)
    # record.flush()
    print('Validation Acc: %f\n'%acc)
    if acc > best_acc:
        best_acc = acc
        print('| Saving Best Model ...')
        save_point = './checkpoint/%s_%s%s_run%d_lr%s_ep%s.pth.tar'%(args.id,
                                                                     args.noise_mode,
                                                                     ''.join(str(args.noise_ratio).split('.')),
                                                                     args.run,
                                                                     ''.join(str(args.lr).split('.')),
                                                                     args.num_epochs)
        save_checkpoint({
            'state_dict': net.state_dict(),
        }, save_point) 

def test():
    global test_acc
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
    acc = 100.*correct/total   
    test_acc = acc
    # record.write('Test Acc: %f\n'%acc)
    print('Test Acc: %f\n'%acc)

if not os.path.exists('./checkpoint'):
    os.mkdir('checkpoint')
# record=open('./checkpoint/'+args.id+'_test.txt','w')
# record.write('learning rate: %f\n'%args.lr)
# record.flush()
print('learning rate: %f\n'%args.lr)

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

best_acc = 0
test_acc = 0
# Model
print('\nModel setup')
print('| Building net')
net = models.PreActResNet32()
test_net = models.PreActResNet32()
if use_cuda:
    net.cuda()
    test_net.cuda()
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

print('\nTraining model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(args.optim_type))

for epoch in range(1, 1+args.num_epochs):
    train(epoch)
    # val(epoch)

print('\nTesting model')
checkpoint = torch.load('./checkpoint/%s_%s%s_run%d_lr%s_ep%s.pth.tar'%(args.id,
                                                                        args.noise_mode,
                                                                        ''.join(str(args.noise_ratio).split('.')),
                                                                        args.run,
                                                                        ''.join(str(args.lr).split('.')),
                                                                        args.num_epochs))
test_net.load_state_dict(checkpoint['state_dict'])
with torch.no_grad():
    test()

print('* Test results : Acc@1 = %.2f%%' %(test_acc))
# record.write('Test Acc: %.2f\n' %test_acc)
# record.flush()
# record.close()
