'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        x += shortcut
        return x

class PreActResNet(nn.Module):
    def __init__(self):
        super(PreActResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block=PreActBlock, planes=32, num_blocks=5, stride=1)
        self.layer2 = self._make_layer(block=PreActBlock, planes=64, num_blocks=5, stride=2)
        self.layer3 = self._make_layer(block=PreActBlock, planes=128, num_blocks=5, stride=2)
        self.bn_final = nn.BatchNorm2d(128)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(128, 10)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # strides = [stride, 1, 1, 1, 1]
        layers = []
        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, weights=None, get_feat=None, is_training=True):
        if weights==None:
            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = F.relu(self.bn_final(x))
            x = self.avgpool(x)
            feat = x.view(x.size(0), -1)
            out = self.linear(feat)
            if get_feat:
                return out,feat
            else:
                return out
        else:
            x = F.conv2d(x, weights['conv1.weight'], stride=1, padding=1)
            #layer 1
            strides = [1, 1, 1, 1, 1]
            for i in range(5):
                if 'layer1.%d.shortcut.0.weight'%i in weights.keys():
                    shortcut = F.conv2d(x, weights['layer1.%d.shortcut.0.weight'%i], stride=strides[i])
                else:
                    shortcut = x
                x = F.batch_norm(x, self.layer1[i].bn1.running_mean, self.layer1[i].bn1.running_var, 
                                 weights['layer1.%d.bn1.weight'%i], weights['layer1.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer1.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer1[i].bn2.running_mean, self.layer1[i].bn2.running_var, 
                                 weights['layer1.%d.bn2.weight'%i], weights['layer1.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer1.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + shortcut
            #layer 2
            strides = [2, 1, 1, 1, 1]
            for i in range(5):
                if 'layer2.%d.shortcut.0.weight'%i in weights.keys():
                    shortcut = F.conv2d(x, weights['layer2.%d.shortcut.0.weight'%i], stride=strides[i])
                else:
                    shortcut = x
                x = F.batch_norm(x, self.layer2[i].bn1.running_mean, self.layer2[i].bn1.running_var, 
                                 weights['layer2.%d.bn1.weight'%i], weights['layer2.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer2.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer2[i].bn2.running_mean, self.layer2[i].bn2.running_var, 
                                 weights['layer2.%d.bn2.weight'%i], weights['layer2.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer2.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + shortcut
            #layer 3
            strides = [2, 1, 1, 1, 1]
            for i in range(5):
                if 'layer3.%d.shortcut.0.weight'%i in weights.keys():
                    shortcut = F.conv2d(x, weights['layer3.%d.shortcut.0.weight'%i], stride=strides[i])
                else:
                    shortcut = x
                x = F.batch_norm(x, self.layer3[i].bn1.running_mean, self.layer3[i].bn1.running_var, 
                                 weights['layer3.%d.bn1.weight'%i], weights['layer3.%d.bn1.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer3.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer3[i].bn2.running_mean, self.layer3[i].bn2.running_var, 
                                 weights['layer3.%d.bn2.weight'%i], weights['layer3.%d.bn2.bias'%i], training=is_training)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer3.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + shortcut
            x = F.batch_norm(x, self.bn_final.running_mean, self.bn_final.running_var,
                             weights['bn_final.weight'], weights['bn_final.bias'], training=is_training)            
            x = F.threshold(x, 0, 0, inplace=True)
            x = F.avg_pool2d(x, kernel_size=8, stride=1, padding=0)
            feat = x.view(x.size(0), -1)
            out = F.linear(feat, weights['linear.weight'], weights['linear.bias'])                
            return out
    
    def save_BN_state_dict(self):
        bn_state = {}
        for name, param in self.state_dict().items():
            if 'bn' in name and 'running' in name:
                # print('extracting %s...' % name)
                bn_state[name] = torch.randn_like(param)
                bn_state[name].copy_(param)
        return bn_state
    
    def load_BN_state_dict(self, state_dict): 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if 'bn' in name and 'running' in name:
                # print('loading %s...' % name)
                own_state[name].copy_(param)

class PreActBlockBN(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlockBN, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(planes)

class PreActResNetBN(nn.Module):
    def __init__(self):
        super(PreActResNetBN, self).__init__()
        self.in_planes = 32

        self.layer1 = self._make_layer(block=PreActBlockBN, planes=32, num_blocks=5, stride=1)
        self.layer2 = self._make_layer(block=PreActBlockBN, planes=64, num_blocks=5, stride=2)
        self.layer3 = self._make_layer(block=PreActBlockBN, planes=128, num_blocks=5, stride=2)
        self.bn_final = nn.BatchNorm2d(128)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # strides = [stride, 1, 1, 1, 1]
        layers = []
        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def load_my_state_dict(self, state_dict): 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)


def PreActResNet32():
    return PreActResNet()

def PreActResNet32BN():
    return PreActResNetBN()
