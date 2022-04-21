from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch

            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, noise_ratio, noise_mode, root_dir, transform, mode, train_val_split_file='', noise_file='', keep_idx=[]):
        self.noise_ratio = noise_ratio
        self.transform = transform
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        self.transition2 = {0:7,1:1,2:2,3:1,4:4,5:5,6:5,7:0,8:2,9:9} # unnatural class transition for checking affiliation matrix (airplane --> horse)
        if self.mode=='test':
            if dataset=='cifar10':
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']
        else: # 'train' or 'val', load the original training data (or 'train_val' for feature extraction)
            self.train_data_all = []
            self.train_label_all = []
            if dataset=='cifar10':
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    self.train_data_all.append(data_dic['data'])
                    self.train_label_all = self.train_label_all+data_dic['labels']
                self.train_data_all = np.concatenate(self.train_data_all)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                self.train_data_all = train_dic['data']
                self.train_label_all = train_dic['fine_labels']
            self.train_data_all = self.train_data_all.reshape((50000, 3, 32, 32))
            self.train_data_all = self.train_data_all.transpose((0, 2, 3, 1))
            
            if not self.mode=='train_val':
                # split the original training data into 'train' and 'val' by a 9:1 ratio
                if os.path.exists(train_val_split_file):
                    print('read saved train_val_split_file (%s)' % self.mode)
                    train_val_split = json.load(open(train_val_split_file,"r"))
                    train_idx = train_val_split['train_idx']
                    val_idx = train_val_split['val_idx']
                else:
                    print('save new train_val_split_file')
                    idx_all = list(range(50000))
                    np.random.shuffle(idx_all)
                    train_idx = idx_all[:45000] # first 45000
                    val_idx = sorted(idx_all[-5000:]) # last 5000, sorted
                    train_val_split = {}
                    train_val_split['train_idx'] = train_idx
                    train_val_split['val_idx'] = val_idx
                    json.dump(train_val_split,open(train_val_split_file,'w'))
                if self.mode=='train':
                    self.train_data = self.train_data_all[train_idx]
                    self.train_label = [self.train_label_all[i] for i in train_idx]
                    if os.path.exists(noise_file):
                        print('read saved noise_file')
                        self.noise_label = json.load(open(noise_file,'r'))
                    else:
                        print('save new noise_file')
                        if noise_mode in ['pseudo','pseu']:
                            raise NotImplementedError('%s does not exist, create it using create_pseudo.py' % noise_file)
                        else:
                            self.noise_label = []
                            idx_train = list(range(45000))
                            random.shuffle(idx_train)
                            num_noise = int(self.noise_ratio*45000)
                            noise_idx = idx_train[:num_noise]
                            for i in range(45000):
                                if i in noise_idx:
                                    if noise_mode=='sym':
                                        if dataset=='cifar10':
                                            self.noise_label.append(np.random.randint(0,9))
                                        elif dataset=='cifar100':
                                            self.noise_label.append(np.random.randint(0,99))
                                    elif noise_mode=='asym':
                                        # only defined for cifar-10
                                        self.noise_label.append(self.transition[self.train_label[i]])
                                    elif noise_mode=='unnat':
                                        # only defined for cifar-10
                                        self.noise_label.append(self.transition2[self.train_label[i]])
                                else:
                                    self.noise_label.append(self.train_label[i])
                            json.dump(self.noise_label,open(noise_file,'w'))
                    # sample selection based on keep_idx (if it is not an empty list)
                    if len(keep_idx) > 0:
                        self.train_data = self.train_data[keep_idx]
                        self.noise_label = [self.noise_label[idx] for idx in keep_idx]
                else:
                    self.val_data = self.train_data_all[val_idx]
                    self.val_label = [self.train_label_all[i] for i in val_idx]

    def __getitem__(self, index):
        if self.mode=='train_val':
            img, target = self.train_data_all[index], self.train_label_all[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
        elif self.mode=='train':
            img, target, target_real = self.train_data[index], self.noise_label[index], self.train_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, target_real, index
        elif self.mode=='val':
            img, target = self.val_data[index], self.val_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode=='train_val':
            return len(self.train_data_all)
        elif self.mode=='train':
            return len(self.train_data)
        elif self.mode=='val':
            return len(self.val_data)
        elif self.mode=='test':
            return len(self.test_data)

class cifar_dataloader():  
    def __init__(self, dataset, noise_ratio, noise_mode, batch_size, num_workers, root_dir, log='', train_val_split_file='', noise_file='', ext_feat=False, sample_filtering=False, keep_idx=[]):
        self.dataset = dataset
        self.noise_ratio = noise_ratio
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.train_val_split_file = train_val_split_file
        self.noise_file = noise_file
        self.ext_feat = ext_feat
        if self.ext_feat:
            self.transform_resnet = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                ]) 
        else:
            if self.dataset=='cifar10':
                self.transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                    ]) 
                self.transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                    ])    
            elif self.dataset=='cifar100':    
                self.transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                    ]) 
                self.transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                    ])
        self.sample_filtering = sample_filtering
        self.keep_idx = keep_idx
    def run(self,):
        if self.ext_feat:
            train_val_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, noise_ratio=self.noise_ratio, root_dir=self.root_dir, transform=self.transform_resnet, mode="train_val")
            train_val_loader = DataLoader(
                dataset=train_val_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)             
            return train_val_loader
        elif self.sample_filtering:
            train_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, noise_ratio=self.noise_ratio, root_dir=self.root_dir, transform=self.transform_train, mode="train", train_val_split_file=self.train_val_split_file, noise_file=self.noise_file)
            train_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return train_loader
        else:
            train_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, noise_ratio=self.noise_ratio, root_dir=self.root_dir, transform=self.transform_train, mode="train", train_val_split_file=self.train_val_split_file, noise_file=self.noise_file, keep_idx=self.keep_idx)
            val_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, noise_ratio=self.noise_ratio, root_dir=self.root_dir, transform=self.transform_test, mode="val",train_val_split_file=self.train_val_split_file)
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, noise_ratio=self.noise_ratio, root_dir=self.root_dir, transform=self.transform_test, mode='test')
            train_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)             
            val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return train_loader, val_loader, test_loader
