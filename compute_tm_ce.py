import sys
import numpy as np
import re
import glob
import os

log_file = sys.argv[1]
find_args = re.search('rw[0-9]Scratch_([a-z]+)([0-9]+)_run([0-9]+).+_log', log_file)
noise_mode = find_args.group(1)
noise_ratio_str = find_args.group(2)
run = int(find_args.group(3))

def filename_to_float(string):
    r = 0
    for i in range(len(string)):
        if string[i] == '0':
            r += 1
        else:
            break
    if r == len(string):
        return 0.0
    else:
        return float(string[r:]) / 10**r

noise_ratio = filename_to_float(noise_ratio_str)

class_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_class = len(class_list)

if noise_mode == 'sym':
    tm_gt = np.full((num_class, num_class), fill_value=noise_ratio/num_class)
    for i in range(num_class):
        tm_gt[i][i] = tm_gt[i][i] + (1-noise_ratio)
elif noise_mode == 'asym' or noise_mode == 'unnat':
    if noise_mode == 'asym':
        transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8}
    elif noise_mode == 'unnat':
        transition = {0:7,1:1,2:2,3:1,4:4,5:5,6:5,7:0,8:2,9:9}
    tm_gt = np.full((num_class, num_class), fill_value=0.0)
    for i in range(num_class):
        tm_gt[i][i] = tm_gt[i][i] + (1-noise_ratio)
        tm_gt[i][transition[i]] = tm_gt[i][transition[i]] + noise_ratio
elif noise_mode == 'pseu' or noise_mode == 'pseudo':
    noise_file = os.path.join('/data1/cclin/cifar-10-batches-py', '%s00_run%d.json' % (noise_mode, run))
    noise_label = json.load(open(noise_file, 'r'))
    targets_train = json.load(open('/data1/cclin/cifar-10-batches-py/sym00_run0.json'), 'r'))
    tm_gt = np.zeros((num_class, num_class))
    for i in range(len(targets_train)):
        tm_gt[targets_train[i]][noise_label[i]] += 1
    row_sums = tm.sum(axis=1, keepdims=True)
    tm_gt = tm_gt / row_sums

tm_list = []
for file in glob.glob(os.path.join('checkpoint', log_file.replace('_log', '_tm_ite*.npy'))):
    tm_list.append(file)

counter = 0
for tm_path in sorted(tm_list):
    counter += 1
    tm = np.load(tm_path)
    trace = np.trace(tm)
    tm_ce = np.multiply(tm_gt, -np.log(tm)).mean()
    print('for ite%d, tm_ce=%f, trace=%f' % (counter, tm_ce, trace))
