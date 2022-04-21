from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import models.preact_resnetBN as models

import os
import sys
import time
import argparse
import datetime

from torch.autograd import Variable

import dataloader_rw as dataloader

from collections import OrderedDict
import math
import random
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import kde

from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='PyTorch Clothing-1M Training')
parser.add_argument('--lr', default=0.2, type=float, help='learning_rate')
parser.add_argument('--lr_rw', default=0.2, type=float, help='learning_rate for sample reweighting')
parser.add_argument('--fast_lr', default=0.2, type=float, help='meta learning_rate for E-steps (model training)')
parser.add_argument('--fast_lr_rw', default=0.01, type=float, help='meta learning_rate for M-steps (sample reweighting)')
parser.add_argument('--num_ssl', default=10, type=int, help='number of intentional perturbations for SSL')
parser.add_argument('--num_rw', default=10, type=int, help='number of intentional perturbations for sample reweighting')
parser.add_argument('--rampup_epoch', default=20, type=int) # "... ramp up eta (meta-learning rate) from 0 to 0.4 during the first 20 epochs"
parser.add_argument('--lrdecay_epoch', default=0, type=int) # "For each training iteration, we divide the learning rate by 10 after 80 epochs, and train until 120 epochs"
parser.add_argument('--lrdecay_factor', default=1.0, type=float)
parser.add_argument('--kl_epoch', default=20, type=int) # Epoch to start to record KL-divergence
parser.add_argument('--num_epoch_warmup', default=0, type=int)
parser.add_argument('--num_iteration', default=2, type=int)
parser.add_argument('--train_per_iter', default=1, type=int)
parser.add_argument('--rw_per_iter', default=1, type=int)

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--meta_lr', default=0.4, type=float)
parser.add_argument('--gamma_init', default=0.99, type=float, help='Initial exponential moving average weight for the teacher model')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--id', default='emB')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--noise_ratio', default=0.5, type=float, help='noise ratio')
parser.add_argument('--noise_mode', default='sym', help='sym or asym')
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
parser.add_argument('--run', default=0, type=int, help='run id (0, 1, 2, 3, or 4) to specify the version of noisy labels')
parser.add_argument('--T', default=10.0, type=float, help='Inverse temperature for the weight')
parser.add_argument('--partial_kl', action='store_true', help='Use partial KL (masked) or full KL (unmasked) to estimate the class transition matrix')
parser.add_argument('--sharpen', default=20, type=int, help='The power to raise the KL statistics to create the estimated transition matrix')
parser.add_argument('--sw_file', default='sw.npy')
parser.add_argument('--tch_file', default='model.pth.tar')
parser.add_argument('--tm_keep_r', default=0.99, type=float, help='EMA keep rate for the estimated transition matrix')
parser.add_argument('--diag_multi', default=10.0, type=float, help='multiplier for the loss of diagonal')
parser.add_argument('--inv_off_diag', action='store_true', help='Use 1/loss_off_diag or -loss_off_diag')
parser.add_argument('--skip_final', action='store_true', help='Skip the final E-step if present')
parser.add_argument('--reinit', action='store_true', help='Reinitialize the model and the estimated class transition matrix for each E-step')
args = parser.parse_args()

random.seed(args.seed)
torch.cuda.set_device(args.gpuid)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_cuda = torch.cuda.is_available()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# Sample reweighting
def reweighting(epoch):
    print('\n=> %s Reweighting Epoch #%d' %(args.id, epoch))
    bn_state = mentor_net.save_BN_state_dict()
    for batch_idx, (inputs, targets, targets_real, sample_idx) in enumerate(train_loader):
        if use_cuda:
            inputs, targets, sample_idx = inputs.cuda(), targets.cuda(), sample_idx.cuda()
        optimizer_rw.zero_grad()
        mentor_net.eval()
        mentor_outputs_base = mentor_net(inputs) # BN in evaluation mode: use running statistics and do not update them
        p_tch = F.softmax(mentor_outputs_base, dim=1)
        p_tch = p_tch.detach()
        mentor_net.train()
        if False and batch_idx == 0:
            print('Before mentor_outputs = mentor_net(inputs)')
            print('mentor_net.layer1[0].bn1.running_mean:')
            print(mentor_net.layer1[0].bn1.running_mean)
            for name, param in mentor_net.named_parameters():
                if 'layer1.1.bn1' in name:
                    print('(mentor_net)', name, param.data[0:3])
        mentor_outputs = mentor_net(inputs) # BN in training mode: use batch statistics and update running statistics
        if False and batch_idx == 0:
            print('After mentor_outputs = mentor_net(inputs)')
            print('mentor_net.layer1[0].bn1.running_mean:')
            print(mentor_net.layer1[0].bn1.running_mean)
            for name, param in mentor_net.named_parameters():
                if 'layer1.1.bn1' in name:
                    print('(mentor_net)', name, param.data[0:3])
        
        if False and batch_idx == 0:
            mentor_net.eval()
            val_loss2 = 0
            correct2 = 0
            total2 = 0
            for batch_idx2, (inputs2, targets2) in enumerate(val_loader):
                if use_cuda:
                    inputs2, targets2 = inputs2.cuda(), targets2.cuda()
                outputs2 = mentor_net(inputs2)
                loss2 = criterion(outputs2, targets2)
                val_loss2 += loss2.data.item()
                _, predicted2 = torch.max(outputs2.data, 1)
                total2 += targets2.size(0)
                correct2 += predicted2.eq(targets2.data).cpu().sum()
            acc2 = 100.*correct2/total2
            print("Before mentor_net.forward(), Validation Loss: %.4f Acc@1: %.2f%%" %(loss2.data.item(), acc2))
            print('mentor_net.layer1[0].bn1.running_mean:')
            print(mentor_net.layer1[0].bn1.running_mean)
            for name, param in mentor_net.named_parameters():
                if 'layer1.1.bn1' in name:
                    print('(mentor_net)', name, param.data[0:3])
            mentor_net.train()
        
        loss_weights = torch.sigmoid(sample_weights.index_select(dim=0, index=sample_idx) * args.T)
        if False and batch_idx == 0:
            print('loss_weights:', loss_weights)
        
        targets_fast = targets.clone()
        fast_loss = torch.matmul(criterion_rw(mentor_outputs, targets_fast), loss_weights)
        if False and batch_idx == 0:
            print('fast_loss:', fast_loss)
        grads = torch.autograd.grad(fast_loss, mentor_net.parameters(), create_graph=True)
        # grad_counter = 0
        # for grad in grads:
        #     if batch_idx == 0 and grad_counter == 0:
        #         print(grad)
        #         grad_counter += 1
        #     grad = grad.detach()
        #     grad.requires_grad = False
        fast_weights = OrderedDict((name, param - args.fast_lr_rw*grad) for ((name, param), grad) in zip(mentor_net.named_parameters(), grads))
        fast_out = mentor_net.forward(x=inputs, weights=fast_weights, get_feat=None, is_training=False)
        
        if False and batch_idx == 0:
            mentor_net.eval()
            val_loss2 = 0
            correct2 = 0
            total2 = 0
            for batch_idx2, (inputs2, targets2) in enumerate(val_loader):
                if use_cuda:
                    inputs2, targets2 = inputs2.cuda(), targets2.cuda()
                outputs2 = mentor_net(inputs2)
                loss2 = criterion(outputs2, targets2)
                val_loss2 += loss2.data.item()
                _, predicted2 = torch.max(outputs2.data, 1)
                total2 += targets2.size(0)
                correct2 += predicted2.eq(targets2.data).cpu().sum()
            acc2 = 100.*correct2/total2
            print("After mentor_net.forward(), Validation Loss: %.4f Acc@1: %.2f%%" %(loss2.data.item(), acc2))
            print('mentor_net.layer1[0].bn1.running_mean:')
            print(mentor_net.layer1[0].bn1.running_mean)
            for name, param in mentor_net.named_parameters():
                if 'layer1.1.bn1' in name:
                    print('(mentor_net)', name, param.data[0:3])
            mentor_net.train()
        
        logp_fast = F.log_softmax(fast_out,dim=1)
        kl_div_vector = consistent_criterion(logp_fast, p_tch)
        rw_loss_diagonal = torch.mean(kl_div_vector)
        if batch_idx == 0:
            print('rw_loss_diagonal (before amplify):', rw_loss_diagonal)
        rw_loss_diagonal *= args.diag_multi
        if False and batch_idx == 0:
            with torch.no_grad():
                w_before_grads = [w.grad for w in sample_weights.index_select(dim=0, index=sample_idx)]
                print('w_before_grads:', w_before_grads)
        rw_loss_diagonal.backward()
        mentor_net.load_BN_state_dict(bn_state)
        if False and batch_idx == 0:
            print('After mentor_net.load_BN_state_dict(...)')
            print('mentor_net.layer1[0].bn1.running_mean:')
            print(mentor_net.layer1[0].bn1.running_mean)
            for name, param in mentor_net.named_parameters():
                if 'layer1.1.bn1' in name:
                    print('(mentor_net)', name, param.data[0:3])
        if False and batch_idx == 0:
            with torch.no_grad():
                w_after_grads = [w.grad for w in sample_weights.index_select(dim=0, index=sample_idx)]
                print('w_after_grads:', w_after_grads)
        if batch_idx == 0:
            print('rw_loss_diagonal (after amplify):', rw_loss_diagonal)
        
        # C1 --> C2, 1-step GD
        for i in range(args.num_rw):
            # if batch_idx == 0:
            #     print('i=%d'%i)
            targets_fast = targets.clone()
            # choose C1 and C2 for all mini-batches in this epoch
            rand_lb_pair = np.random.choice(range(num_class), size=2, replace=False) # note: we want C1 != C2 here
            idx0 = [idx for idx in range(targets.size(0)) if targets[idx] == rand_lb_pair[0]]
            for n in range(targets.size(0)):
                if n in idx0:
                    targets_fast[n] = rand_lb_pair[1]
            # forward again
            mentor_outputs = mentor_net(inputs)
            loss_weights = torch.sigmoid(sample_weights.index_select(dim=0, index=sample_idx) * args.T)
            fast_loss = torch.matmul(criterion_rw(mentor_outputs,targets_fast), loss_weights)
            grads = torch.autograd.grad(fast_loss, mentor_net.parameters(), create_graph=True)
            # for grad in grads:
            #     grad = grad.detach()
            #     grad.requires_grad = False
            fast_weights = OrderedDict((name, param - args.fast_lr_rw*grad) for ((name, param), grad) in zip(mentor_net.named_parameters(), grads))
            fast_out = mentor_net.forward(x=inputs, weights=fast_weights, get_feat=None, is_training=False)
            logp_fast = F.log_softmax(fast_out,dim=1)
            kl_div_vector = consistent_criterion(logp_fast, p_tch)
            kl_div_reduced = torch.mean(kl_div_vector)
            if args.inv_off_diag:
                rw_loss_others = 1/kl_div_reduced
            else:
                rw_loss_others = -kl_div_reduced
            if batch_idx == 0:
                with torch.no_grad():
                    if i == 0:
                        rw_loss_off_diag = kl_div_reduced
                    else:
                        rw_loss_off_diag += kl_div_reduced
            rw_loss_others.backward()
            mentor_net.load_BN_state_dict(bn_state)
        if batch_idx == 0:
            print('rw_loss_off_diag:', rw_loss_off_diag)
            with torch.no_grad():
                print('rw_loss_off_diag/rw_loss_diagonal=%.4f' % torch.true_divide(rw_loss_off_diag, rw_loss_diagonal).data.item())
                print('rw_loss_diagonal/rw_loss_off_diag**2=%.4f' % torch.true_divide(rw_loss_diagonal, rw_loss_off_diag**2).data.item())
        if False and batch_idx == 0:
            with torch.no_grad():
                w_before = torch.sigmoid(sample_weights.index_select(dim=0, index=sample_idx) * args.T)
                print('weights before update:')
                print(w_before)
        optimizer_rw.step()
        if False and batch_idx == 0:
            with torch.no_grad():
                w_after = torch.sigmoid(sample_weights.index_select(dim=0, index=sample_idx) * args.T)
                print('weights after update:')
                print(w_after)

# Training
def train(epoch, lr_decay_power=0, use_mentor=False):
    global init
    global first_ite
    net.train()
    tch_net.train()
    train_loss = 0
    train_loss_ssl = 0
    correct = 0
    total = 0
    
    learning_rate = args.lr*(args.lrdecay_factor**lr_decay_power)
    # divide learning rate only if args.lrdecay_epoch > 0
    if args.lrdecay_epoch > 0:
        if epoch > args.lrdecay_epoch:
            learning_rate=learning_rate/10
        
    if epoch>args.rampup_epoch:
        meta_lr = args.meta_lr
        gamma = 0.999
        tch_r = 0.5
    else:
        u = epoch/args.rampup_epoch
        meta_lr = args.meta_lr*math.exp(-5*(1-u)**2)
        gamma = args.gamma_init
        tch_r = 0.5*math.exp(-5*(1-u)**2)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print('\n=> %s Training Epoch #%d, LR=%.6f, META_LR=%.6f' %(args.id,epoch, learning_rate, meta_lr))

    tm_tensor = torch.Tensor(tm)
    tm_tensor = tm_tensor.cuda()
    tm_tensor = tm_tensor.detach()
    print('tm_tensor:', tm_tensor)
    
    for batch_idx, (inputs, targets, targets_real, sample_idx) in enumerate(train_loader):
        # if batch_idx < 5:
        #     print('sample_idx:', sample_idx)
        if first_ite and epoch == 1:
            clean_indices.extend([sample_idx[i] for i in range(len(targets)) if targets[i] == targets_real[i]])
        if use_cuda:
            inputs, targets, sample_idx = inputs.cuda(), targets.cuda(), sample_idx.cuda()
        loss_weights = sw_sigmoid.index_select(dim=0, index=sample_idx)
        loss_weights = loss_weights.detach()
        if batch_idx == 0:
            print('loss_weights:', loss_weights)
        optimizer.zero_grad()
        
        # [2022/02/20] move to the end of each batch
        #outputs = net(inputs)               # Forward Propagation
        #outputs_prob = softmax_dim1(outputs)
        #outputs_adjust = torch.mm(outputs_prob, tm_tensor)
        #outputs_log_prob = torch.log(outputs_adjust)
        #nll_loss_all = nll_loss(outputs_log_prob, targets)
        #class_loss = torch.matmul(nll_loss_all, loss_weights) / len(targets)
        #class_loss.backward()
        
        bn_state = net.save_BN_state_dict()
        if False and batch_idx == 0:
            print('\n[Check the saved bn_state]')
            for k,v in bn_state.items():
                if 'bn1' in k and 'running_mean' in k:
                    print(v.data[0:3])
        
        # [meta-leargning loop]
        if (not first_ite) or (epoch > 2):
            if init:
                init = False
                for param,param_tch in zip(net.parameters(),tch_net.parameters()):
                    param_tch.data.copy_(param.data)
            else:
                if False and batch_idx == 0:
                    print('\n[Before updating tch_net]')
                    print('tch_net.layer1[0].bn1.running_mean:', tch_net.layer1[0].bn1.running_mean.data[0:3])
                    for name, param in tch_net.named_parameters():
                        if 'layer1.1.bn1.weight' in name:
                            print(name, param.data[0:3])
                
                for param,param_tch in zip(net.parameters(),tch_net.parameters()):
                    param_tch.data.mul_(gamma).add_(alpha=(1-gamma), other=param.data)
                
                if False and batch_idx == 0:
                    print('[After updating tch_net]')
                    print('tch_net.layer1[0].bn1.running_mean:', tch_net.layer1[0].bn1.running_mean.data[0:3])
                    for name, param in tch_net.named_parameters():
                        if 'layer1.1.bn1.weight' in name:
                            print(name, param.data[0:3])
            
            tch_outputs = tch_net(inputs,get_feat=False)
            if use_mentor:
                mentor_outputs = mentor_net(inputs,get_feat=False)
                p_tch = tch_r * F.softmax(tch_outputs, dim=1) + (1 - tch_r) * F.softmax(mentor_outputs, dim=1)
            else:
                p_tch = F.softmax(tch_outputs,dim=1)
            p_tch = p_tch.detach()
            if False and batch_idx == 0:
                print('p_tch:')
                for i in range(targets.size(0)):
                    for j in range(num_class):
                        print('%.8f' % p_tch[i,j].data.item(), end=' ')
                    print()
            
            for i in range(args.num_ssl):
                if False and batch_idx == 0:
                    print('i=%d' % i)
                targets_fast = targets.clone()
                rand_lb_pair = np.random.choice(range(num_class), size=2, replace=True)
                loss_mask = torch.cuda.FloatTensor(num_class).fill_(1.0)
                for idx in rand_lb_pair:
                    loss_mask[idx] = 0.0
                # print('targets:', targets)
                # if batch_idx == 0:
                #     print('rand_lb_pair:', rand_lb_pair)
                # print('loss_mask:', loss_mask)
                idx0 = [idx for idx in range(targets.size(0)) if targets[idx] == rand_lb_pair[0]]
                # idx1 = [idx for idx in range(targets.size(0)) if targets[idx] == rand_lb_pair[1]]
                # print('idx0:', idx0)
                # print('idx1:', idx1)
                for n in range(targets.size(0)):
                    if n in idx0:
                        targets_fast[n] = rand_lb_pair[1]
                    # elif n in idx1:
                    #     targets_fast[n] = rand_lb_pair[0]
                # print('targets_fast:', targets_fast)
                
                if False and batch_idx == 0:
                    print('\n[Before outputs = net(inputs)]')
                    print('net.layer1[0].bn1.running_mean:', net.layer1[0].bn1.running_mean.data[0:3])
                    for name, param in net.named_parameters():
                        if 'layer1.1.bn1.weight' in name:
                            print(name, param.data[0:3])
                
                outputs = net(inputs) # Forward again
                fast_loss = criterion(outputs,targets_fast)
                # outputs_prob = softmax_dim1(outputs)
                # outputs_adjust = torch.mm(outputs_prob, tm_tensor)
                # outputs_log_prob = torch.log(outputs_adjust)
                # nll_loss_all = nll_loss(outputs_log_prob, targets_fast)
                # fast_loss = torch.matmul(nll_loss_all, loss_weights) / len(targets_fast)
                
                if False and batch_idx == 0:
                    print('[After outputs = net(inputs)]')
                    print('net.layer1[0].bn1.running_mean:', net.layer1[0].bn1.running_mean.data[0:3])
                    for name, param in net.named_parameters():
                        if 'layer1.1.bn1.weight' in name:
                            print(name, param.data[0:3])
                
                # grads = torch.autograd.grad(fast_loss, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
                grads = torch.autograd.grad(fast_loss, net.parameters())
                for grad in grads:
                    grad = grad.detach()
                    grad.requires_grad = False
                fast_weights = OrderedDict((name, param - args.fast_lr*grad) for ((name, param), grad) in zip(net.named_parameters(), grads))
                if False and batch_idx == 0:
                    print('fast_weights["layer1.0.bn1.weight"].data[0:3]=', fast_weights['layer1.0.bn1.weight'].data[0:3])
                
                if False and batch_idx == 0:
                    print('[Before net.forward(...)]')
                    print('net.layer1[0].bn1.running_mean:', net.layer1[0].bn1.running_mean.data[0:3])
                    for name, param in net.named_parameters():
                        if 'layer1.1.bn1.weight' in name:
                            print(name, param.data[0:3])
                
                # fast_out = net.forward(inputs,fast_weights)
                fast_out = net.forward(x=inputs, weights=fast_weights, get_feat=None, is_training=True)
                
                if False and batch_idx == 0:
                    print('[After net.forward(...)]')
                    print('net.layer1[0].bn1.running_mean:', net.layer1[0].bn1.running_mean.data[0:3])
                    for name, param in net.named_parameters():
                        if 'layer1.1.bn1.weight' in name:
                            print(name, param.data[0:3])
                
                logp_fast = F.log_softmax(fast_out,dim=1)
                if False and batch_idx == 0:
                    print('logp_fast:')
                    for i in range(targets.size(0)):
                        for j in range(num_class):
                            print('%.8f' % logp_fast[i,j].data.item(), end=' ')
                        print()
                kl_div_vector = consistent_criterion(logp_fast,p_tch)
                kl_div_masked = torch.matmul(kl_div_vector, loss_mask) / num_class
                
                #### choose one from the following two lines (partial KL or full KL?)
                if args.partial_kl:
                    kl_div_reduced = torch.mean(kl_div_masked, dim=0)
                else:
                    kl_div_reduced = torch.mean(kl_div_vector)
                
                # rand_lb_pair_ordered = sorted(rand_lb_pair)
                # rand_lb_pair_tuple = (rand_lb_pair_ordered[0], rand_lb_pair_ordered[1])
                rand_lb_pair_tuple = (rand_lb_pair[0], rand_lb_pair[1])
                if epoch > args.kl_epoch:
                    if rand_lb_pair_tuple in kl_dict.keys():
                        kl_dict[rand_lb_pair_tuple].append(kl_div_reduced.data.item())
                    else:
                        kl_dict[rand_lb_pair_tuple] = [kl_div_reduced.data.item()]
                meta_loss = meta_lr * torch.mean(kl_div_masked, dim=0)
                meta_loss.backward()
                net.load_BN_state_dict(bn_state)
                if False and batch_idx == 0:
                    print('[After net.load_BN_state_dict(bn_state)]')
                    print('net.layer1[0].bn1.running_mean:', net.layer1[0].bn1.running_mean.data[0:3])
                    for name, param in net.named_parameters():
                        if 'layer1.1.bn1.weight' in name:
                            print(name, param.data[0:3])
                
                if False and batch_idx == 0:
                    print('kl_div_reduced:', kl_div_reduced)
                    print('torch.mean(kl_div_masked):', torch.mean(kl_div_masked))
                    print('meta_loss:', meta_loss)
                train_loss_ssl += meta_loss.data.item()
            
        outputs = net(inputs)
        outputs_prob = softmax_dim1(outputs)
        outputs_adjust = torch.mm(outputs_prob, tm_tensor)
        outputs_log_prob = torch.log(outputs_adjust)
        nll_loss_all = nll_loss(outputs_log_prob, targets)
        class_loss = torch.matmul(nll_loss_all, loss_weights) / len(targets)
        class_loss.backward()
        
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
            print('| Epoch %3d Iter[%3d/%3d]\t\tLoss: %.4f LossSSL: %.4f Acc@1: %.3f%%'
                %(epoch, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, train_loss, train_loss_ssl, 100.*correct/total))
        if batch_idx%50==0:
            with torch.no_grad():
                val(epoch,batch_idx,lr_decay_power+1)
                val_tch(epoch,batch_idx,lr_decay_power+1)
            net.train()
            tch_net.train()
            
            
def val(epoch,batch_idx,iteration):
    global best
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        # val_loss += loss.data[0]
        val_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
    # Save checkpoint when best model
    acc = 100.*correct/total
    if acc < 15:
        print('\n[Validation too bad!]')
        print('outputs:', outputs)
        for name, param in net.state_dict().items():
            if 'conv' in name:
                print(name, param.data[0:3,0,0,0])
            elif 'shortcut' in name:
                print(name, param.data[0:3,0,0,0])
            elif 'linear.weight' in name:
                print(name, param.data[0:3,0])
            elif 'linear.bias' in name:
                print(name, param.data[0:3])
            elif 'bn' in name and not 'num_batches_tracked' in name:
                print(name, param.data[0:3])
        print()
    # print("\n| Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, batch_idx, loss.data[0], acc))
    print("\n| Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, batch_idx, val_loss, acc))
    # record.write('Epoch #%d Batch #%3d  Acc: %.2f' %(epoch,batch_idx,acc))
    # print('Epoch #%d Batch #%3d  Acc: %.2f' %(epoch,batch_idx,acc))
    if acc < 15:
        bn_state = net.save_BN_state_dict()
        net.train()
        val_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        print('outputs:', outputs)
        print("| (validation using batch statistics)\t\t\tLoss: %.4f Acc@1: %.2f%%\n" %(val_loss, acc))
        net.load_BN_state_dict(bn_state)
        
    if acc > best:
    # if False:
        best = acc
        print('| Saving Best Model (net)...')
        save_checkpoint({
            'state_dict': net.state_dict(),
            'best_acc': best,
        }, save_point.replace('.pth.tar', '_ite%02d.pth.tar' % iteration))
        np.save(save_point.replace('.pth.tar', '_tm_ite%02d.npy' % iteration), tm)

def val_tch(epoch,batch_idx,iteration):
    global best
    tch_net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = tch_net(inputs)
        loss = criterion(outputs, targets)
        
        # val_loss += loss.data[0]
        val_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
    # Save checkpoint when best model
    acc = 100.*correct/total
    # print("| tch Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%\n" %(epoch, batch_idx, loss.data[0], acc))
    print("| tch Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, batch_idx, val_loss, acc))
    # record.write(' | tchAcc: %.2f\n' %acc)
    # record.flush()
    # print(' | tchAcc: %.2f\n' %acc)
    
    if acc > best:
    # if False:
        best = acc
        print('| Saving Best Model (tchnet)...')
        save_checkpoint({
            'state_dict': tch_net.state_dict(),
            'best_acc': best,
        }, save_point.replace('.pth.tar', '_ite%02d.pth.tar' % iteration))
        np.save(save_point.replace('.pth.tar', '_tm_ite%02d.npy' % iteration), tm)

def test():
    test_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
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
    
def float_to_filename(num):
    split_result = str(num).split('.')
    if split_result[0] == '0' and split_result[1] == '0':
        return '00'
    elif split_result[1] == '0':
        return split_result[0]
    else:
        return ''.join(split_result)

# ===============================================
# record=open('./checkpoint/'+args.id+'.txt','w')
# record.write('learning rate: %f\n'%args.lr)
# record.write('batch size: %f\n'%args.batch_size)
# record.write('start iter: %d\n'%args.start_iter)
# record.write('mid iter: %d\n'%args.mid_iter)
# record.flush()
print('learning rate: %f\n'%args.lr)
print('batch size: %d\n'%args.batch_size)
print('ramp-up end epoch of the meta-learning rate: %d\n'%args.rampup_epoch)
print('LR decay epoch: %d\n'%args.lrdecay_epoch)

save_point = './checkpoint/%s_%s%s_run%d_ite%dt%dr%d_lr%sfast%srw%sdiag%s_T%s_tm%s.pth.tar'%(args.id,
                                                                                             args.noise_mode,
                                                                                             float_to_filename(args.noise_ratio),
                                                                                             args.run,
                                                                                             args.num_iteration,
                                                                                             args.train_per_iter,
                                                                                             args.rw_per_iter,
                                                                                             float_to_filename(args.lr_rw),
                                                                                             float_to_filename(args.fast_lr),
                                                                                             float_to_filename(args.fast_lr_rw),
                                                                                             float_to_filename(args.diag_multi),
                                                                                             float_to_filename(args.T),
                                                                                             float_to_filename(args.tm_keep_r))
if args.lrdecay_factor < 1.0:
    save_point = save_point.replace('.pth.tar', '_decay%s.pth.tar'%float_to_filename(args.lrdecay_factor))
if args.inv_off_diag:
    save_point = save_point.replace('.pth.tar', '_inv.pth.tar')
else:
    save_point = save_point.replace('.pth.tar', '_neg.pth.tar')
if args.reinit:
    save_point = save_point.replace('.pth.tar', '_reinit.pth.tar')

cidx_path = os.path.join(args.data_path, 'cidx_%s%s_run%d.npy' % (args.noise_mode, float_to_filename(args.noise_ratio), args.run))

# Specify the path to save all density plots
density_path = 'density/%s' % os.path.basename(save_point).replace('.pth.tar', '')
if os.path.exists(density_path):
    print('WARNING: density_path: %s already exists' % density_path)
else:
    os.makedirs(density_path)

num_class = 10 # [TODO] remove magic number

# print the ground truth class transition matrix
if args.noise_mode == 'sym':
    tm_gt = np.full((num_class, num_class), fill_value=args.noise_ratio/num_class)
    for i in range(num_class):
        tm_gt[i][i] = tm_gt[i][i] + (1-args.noise_ratio)
elif args.noise_mode == 'asym' or args.noise_mode == 'unnat':
    if args.noise_mode == 'asym':
        transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8}
    elif args.noise_mode == 'unnat':
        transition = {0:7,1:1,2:2,3:1,4:4,5:5,6:5,7:0,8:2,9:9}
    tm_gt = np.full((num_class, num_class), fill_value=0.0)
    for i in range(num_class):
        tm_gt[i][i] = tm_gt[i][i] + (1-args.noise_ratio)
        tm_gt[i][transition[i]] = tm_gt[i][transition[i]] + args.noise_ratio
print('tm_gt:')
for i in range(num_class):
    for j in range(num_class):
        print('%.4f' % tm_gt[i][j], end=' ')
    print()
print()

best = 0
init = True
# Model
print('\nModel setup')
print('| Building net')
net = models.PreActResNet32()
tch_net = models.PreActResNet32()
test_net = models.PreActResNet32()
mentor_net = models.PreActResNet32()

if use_cuda:
    net.cuda()
    tch_net.cuda()
    test_net.cuda()
    cudnn.benchmark = True

for param in tch_net.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
criterion_rw = nn.CrossEntropyLoss(reduction='none')
softmax_dim1 = nn.Softmax(dim=1)
nll_loss = nn.NLLLoss(reduction='none')
consistent_criterion = nn.KLDivLoss(reduction='none')
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

keep_idx = [] # [Note] empty list: keep all training samples
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
                                                                      args.run),
                                     keep_idx=keep_idx)
train_loader,val_loader,test_loader = loader.run()
print('Before sample filtering, len(train_loader.dataset) = %d' % len(train_loader.dataset))

# Initialize all sample weights
sw_path = os.path.join('checkpoint', args.sw_file)
if os.path.exists(sw_path):
    print('load sample weights from %s' % sw_path)
    sw_np = np.load(sw_path)
    sample_weights = torch.from_numpy(sw_np).cuda(device='cuda').requires_grad_(True)
    ite_start = int(re.search('ite([0-9]+)\.npy', args.sw_file).group(1)) + 1
else:
    print('initialize all sample weights')
    sample_weights = torch.zeros(len(train_loader.dataset), requires_grad=True, device='cuda')
    # sample_weights = torch.ones(len(train_loader.dataset), requires_grad=True, device='cuda')
    ite_start = 1
print('ite_start=%d' % ite_start)
optimizer_rw = optim.SGD([sample_weights], lr=args.lr_rw, momentum=0.9, weight_decay=1e-4)

clean_indices = []
tm = np.eye(num_class) # initialize the transition matrix using an identity matrix

first_ite = True
# [Warm-up phase]
start = time.time()
for epoch in range(1, 1+args.num_epoch_warmup):
    train(epoch, use_mentor=False)
    first_ite = False
    # debug: record which samples are clean and which are noisy
    if epoch == 1:
        if os.path.exists(cidx_path):
            print('load clean_indices from %s' % cidx_path)
            clean_indices = np.load(cidx_path)
        else:
            clean_indices = np.array(clean_indices)
            print('save clean_indices to %s' % cidx_path)
            np.save(cidx_path, clean_indices)
        print('clean_indices.shape:', clean_indices.shape)
        noisy_indices = [i for i in range(len(train_loader.dataset)) if not i in clean_indices]
        noisy_indices = np.array(noisy_indices)
        print('noisy_indices.shape:', noisy_indices.shape)
        is_clean = [1 if i in clean_indices else 0 for i in range(len(train_loader.dataset))]
        print('np.sum(is_clean):', np.sum(is_clean))
    #if epoch == args.num_epoch_warmup:
    #    print('| Saving Last Model (tchnet)...')
    #    save_checkpoint({
    #        'state_dict': tch_net.state_dict()
    #    }, save_point)
print('time elapsed:', time.time() - start)

# [EM phase]
for iteration in range(ite_start, ite_start+args.num_iteration):
    start = time.time()
    # [E-step]
    print('[TRAINING MODE]')
    # [re-initialize both the student and the teacher models]
    if args.reinit:
        if iteration > ite_start:
            for layer in net.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            for layer in tch_net.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        tm = np.eye(num_class) # re-initialize the transition matrix using an identity matrix
    # [normalize all sample weights]
    with torch.no_grad():
        sw_sigmoid = torch.sigmoid(sample_weights * args.T)
        sw_sigmoid_mean = torch.mean(sw_sigmoid)
        print('before normalization, sw_sigmoid_mean=%.4f' % sw_sigmoid_mean.data.item())
        sw_sigmoid /= sw_sigmoid_mean
        sw_sigmoid_mean = torch.mean(sw_sigmoid)
        print('after normalization, sw_sigmoid_mean=%.4f' % sw_sigmoid_mean.data.item())
    tch_path = os.path.join('checkpoint', args.tch_file)
    if iteration == ite_start and os.path.exists(tch_path) and os.path.exists(cidx_path):
        last_model = torch.load(tch_path)
        clean_indices = np.load(cidx_path)
        print('clean_indices.shape:', clean_indices.shape)
        noisy_indices = [i for i in range(len(train_loader.dataset)) if not i in clean_indices]
        noisy_indices = np.array(noisy_indices)
        print('noisy_indices.shape:', noisy_indices.shape)
        is_clean = [1 if i in clean_indices else 0 for i in range(len(train_loader.dataset))]
        print('np.sum(is_clean):', np.sum(is_clean))
        first_ite = False
    else:
        for ep in range(1, 1+args.train_per_iter):
            kl_dict = {}
            train(ep, lr_decay_power=iteration-1, use_mentor=False)
            first_ite = False
            # debug: record which samples are clean and which are noisy
            if iteration == ite_start and ep == 1:
                if os.path.exists(cidx_path):
                    print('load clean_indices from %s' % cidx_path)
                    clean_indices = np.load(cidx_path)
                else:
                    clean_indices = np.array(clean_indices)
                    print('save clean_indices to %s' % cidx_path)
                    np.save(cidx_path, clean_indices)
                print('clean_indices.shape:', clean_indices.shape)
                noisy_indices = [i for i in range(len(train_loader.dataset)) if not i in clean_indices]
                noisy_indices = np.array(noisy_indices)
                print('noisy_indices.shape:', noisy_indices.shape)
                is_clean = [1 if i in clean_indices else 0 for i in range(len(train_loader.dataset))]
                print('np.sum(is_clean):', np.sum(is_clean))
            if ep > args.kl_epoch:
                # debug: check min/max number of ordered class pairs
                # print('len(kl_dict.keys()):', len(kl_dict.keys()))
                # min_n_kl = 3520
                # max_n_kl = 0
                # for k, v in kl_dict.items():
                #     if len(v) < min_n_kl:
                #         min_n_kl = len(v)
                #     if len(v) > max_n_kl:
                #         max_n_kl = len(v)
                # print('min_n_kl=%d, max_n_kl=%d' % (min_n_kl, max_n_kl))
                tm_from_kl = np.zeros([num_class,num_class])
                for i in range(num_class):
                    for j in range(num_class):
                        if (i, j) in kl_dict.keys():
                            tm_from_kl[i][j] = np.mean(kl_dict[(i,j)])
                tm_from_kl = np.power(tm_from_kl+1e-10, -float(args.sharpen))
                row_sums = tm_from_kl.sum(axis=1, keepdims=True)
                tm_from_kl = tm_from_kl / row_sums
                tm = args.tm_keep_r * tm + (1 - args.tm_keep_r) * tm_from_kl
                trace = np.trace(tm)
                tm_ce = np.multiply(tm_gt, -np.log(tm)).mean()
                print('at ite%02dep%03d, tm_ce = %.6f, trace = %.6f' % (iteration, ep, tm_ce, trace))
        # Run testing using the last model of each E-step
        #print('| Saving Last Model (tchnet)...')
        #save_checkpoint({
        #    'state_dict': tch_net.state_dict()
        #}, save_point.replace('.pth.tar', '_ite%02d.pth.tar' % iteration))
        #np.save(save_point.replace('.pth.tar', '_tm_ite%02d.npy' % iteration), tm)
        print('tm:')
        for i in range(num_class):
            for j in range(num_class):
                print('%.4f' % tm[i][j], end=' ')
            print()
        print('\nTesting model at iteration %d' % iteration)
        last_model = torch.load(save_point.replace('.pth.tar', '_ite%02d.pth.tar' % iteration))
    test_net.load_state_dict(last_model['state_dict'])
    with torch.no_grad():
        test()
    print('time elapsed:', time.time() - start)
    # [M-step]
    with torch.no_grad():
        w_temp = torch.sigmoid(sample_weights * args.T)
        # w_temp = sample_weights
        w_temp_clean = w_temp[clean_indices]
        w_temp_noisy = w_temp[noisy_indices]
        print('(use initialized sample_weights to compute the following statistics)')
        print('average weights for clean samples:', torch.mean(w_temp_clean))
        print('average weights for noisy samples:', torch.mean(w_temp_noisy))
        print('clean/noisy AUC: %.4f' % roc_auc_score(is_clean, sample_weights.cpu().numpy()))
    start = time.time()
    # with torch.no_grad():
    #     for i in range(len(train_loader.dataset)):
    #         sample_weights[i] = 0.0
    if iteration == ite_start and os.path.exists(tch_path):
        mentor_ckpt = torch.load(tch_path)
        print('load mentor model from %s' % tch_path)
    else:
        mentor_ckpt = torch.load(save_point.replace('.pth.tar', '_ite%02d.pth.tar' % iteration))
        print('load mentor model from %s' % save_point.replace('.pth.tar', '_ite%02d.pth.tar' % iteration))
    mentor_net.load_state_dict(mentor_ckpt['state_dict'])
    mentor_net.cuda()
    mentor_net.train()
    print('[REWEIGHTING MODE]')
    for ep in range(1, 1+args.rw_per_iter):
        reweighting(ep)
        # debug: make sure all model weights are fixed during reweighting phase
        # print('the first 10 model weights in the last layer: ', end='')
        # for name, param in mentor_net.named_parameters():
        #     if name == 'linear.weight':
        #        print(param[0][0:10])
               #for i in range(10):
               #    print('%.4f' % param[i].data.item(), end='|')
               #print()
        if True or ep == args.rw_per_iter:
            with torch.no_grad():
                w_temp = torch.sigmoid(sample_weights * args.T)
                # w_temp = sample_weights
                w_temp_clean = w_temp[clean_indices]
                w_temp_noisy = w_temp[noisy_indices]
                print('average weights for clean samples:', torch.mean(w_temp_clean))
                print('average weights for noisy samples:', torch.mean(w_temp_noisy))
                x = np.linspace(0,1,100)
                w_temp_clean_nparray = w_temp_clean.cpu().numpy()
                print('w_temp_clean_nparray.shape:', w_temp_clean_nparray.shape)
                print('w_temp_clean_nparray[0]:', w_temp_clean_nparray[0])
                counter = 0
                for i in range(w_temp_clean_nparray.shape[0]):
                    if w_temp_clean_nparray[i] == 0.5:
                        counter += 1
                print('counter=%d' % counter)
                density_clean = kde.gaussian_kde(w_temp_clean.cpu().numpy())
                y_clean = density_clean(x)
                density_noisy = kde.gaussian_kde(w_temp_noisy.cpu().numpy())
                y_noisy = density_noisy(x)
                plt.plot(x, y_clean)
                plt.plot(x, y_noisy)
                plt.savefig(os.path.join(density_path, 'ite%02d_ep%03d.png' % (iteration, ep)))
                plt.close()
                print('clean/noisy AUC: %.4f' % roc_auc_score(is_clean, sample_weights.cpu().numpy()))
    print('time elapsed:', time.time() - start)
    np.save(save_point.replace('.pth.tar', '_sw_ite%02d.npy' % iteration), sample_weights.detach().cpu().numpy())
# [the final E-step]
if not args.skip_final:
    iteration = ite_start+args.num_iteration 
    start = time.time()
    #for layer in net.children():
    #    if hasattr(layer, 'reset_parameters'):
    #        layer.reset_parameters()
    #for layer in tch_net.children():
    #    if hasattr(layer, 'reset_parameters'):
    #        layer.reset_parameters()
    #tm = np.eye(num_class) # re-initialize the transition matrix using an identity matrix
    with torch.no_grad():
        sw_sigmoid = torch.sigmoid(sample_weights * args.T)
        sw_sigmoid_mean = torch.mean(sw_sigmoid)
        print('before normalization, sw_sigmoid_mean=%.4f' % sw_sigmoid_mean.data.item())
        sw_sigmoid /= sw_sigmoid_mean
        sw_sigmoid_mean = torch.mean(sw_sigmoid)
        print('after normalization, sw_sigmoid_mean=%.4f' % sw_sigmoid_mean.data.item())
    print('[TRAINING MODE]')
    for ep in range(1, 1+args.train_per_iter):
        # tm_keep_r = 0.99
        kl_dict = {}
        train(ep, lr_decay_power=iteration-1, use_mentor=False)
        # debug: make sure all samples weights are fixed during training phase
        # print('the first 10 sample weights: ', end='')
        # for i in range(10):
        #     print('%.4f' % sample_weights[i].data.item(), end='')
        #     if i < 9:
        #         print('|', end='')
        # print()
        if ep > args.kl_epoch:
            print('len(kl_dict.keys()):', len(kl_dict.keys()))
            min_n_kl = 3520
            max_n_kl = 0
            for k, v in kl_dict.items():
                if len(v) < min_n_kl:
                    min_n_kl = len(v)
                if len(v) > max_n_kl:
                    max_n_kl = len(v)
            print('min_n_kl=%d, max_n_kl=%d' % (min_n_kl, max_n_kl))
            tm_from_kl = np.zeros([num_class,num_class])
            for i in range(num_class):
                for j in range(num_class):
                    if (i, j) in kl_dict.keys():
                        tm_from_kl[i][j] = np.mean(kl_dict[(i,j)])
            tm_from_kl = np.power(tm_from_kl+1e-10, -float(args.sharpen))
            row_sums = tm_from_kl.sum(axis=1, keepdims=True)
            tm_from_kl = tm_from_kl / row_sums
            tm = args.tm_keep_r * tm + (1 - args.tm_keep_r) * tm_from_kl
            trace = np.trace(tm)
            tm_ce = np.multiply(tm_gt, -np.log(tm)).mean()
            print('at ite%02dep%03d, tm_ce = %.6f, trace = %.6f' % (iteration, ep, tm_ce, trace))
    # Run testing using the last model of each E-step
    #print('| Saving Last Model (tchnet)...')
    #save_checkpoint({
    #    'state_dict': tch_net.state_dict()
    #}, save_point.replace('.pth.tar', '_ite%02d.pth.tar' % iteration))
    #np.save(save_point.replace('.pth.tar', '_tm_ite%02d.npy' % iteration), tm)
    print('tm:')
    for i in range(num_class):
        for j in range(num_class):
            print('%.4f' % tm[i][j], end=' ')
        print()
    print('\nTesting model at iteration %d' % iteration)
    last_model = torch.load(save_point.replace('.pth.tar', '_ite%02d.pth.tar' % iteration))
    test_net.load_state_dict(last_model['state_dict'])
    with torch.no_grad():
        test()
    print('time elapsed:', time.time() - start)


# Run testing only once using the best model
# print('\nTesting model')
# best_model = torch.load(save_point)
# test_net.load_state_dict(best_model['state_dict'])
# with torch.no_grad():
#     test()


# record.close()
