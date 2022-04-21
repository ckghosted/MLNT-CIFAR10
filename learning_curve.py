import sys, re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

log_fname = sys.argv[1]
learning_curve_fname = log_fname.replace('_log', '_lc.png')
print('producing %s...' % learning_curve_fname)

with open(log_fname, 'r') as fhand:
    loss_train = []
    loss_val = []
    loss_val_tch = []
    acc_train = []
    acc_val = []
    acc_val_tch = []
    ep_counter = 1
    for line in fhand:
        find_train = re.search('Iter\[  1/[0-9]+\].*Loss: ([0-9]*\.[0-9]*).*Acc@1: ([0-9]*\.[0-9]*)\%', line)
        if find_train:
            loss_train.append(float(find_train.group(1)))
            acc_train.append(float(find_train.group(2)))
        find_val = re.search('\| Validation.*#  0.*Loss: ([0-9]*\.[0-9]*) Acc@1: ([0-9]*\.[0-9]*)\%', line)
        find_val_for_baseline = re.search('\| Validation Epoch #([0-9]*)\s*Loss: ([0-9]*\.[0-9]*) Acc@1: ([0-9]*\.[0-9]*)\%', line)
        if find_val:
            loss_val.append(float(find_val.group(1)))
            acc_val.append(float(find_val.group(2)))
        elif find_val_for_baseline:
            if ep_counter == int(find_val_for_baseline.group(1)):
                loss_val.append(float(find_val_for_baseline.group(2)))
                acc_val.append(float(find_val_for_baseline.group(3)))
                ep_counter = ep_counter + 1
        find_val_tch = re.search('\| tch Validation.*#  0.*Loss: ([0-9]*\.[0-9]*) Acc@1: ([0-9]*\.[0-9]*)\%', line)
        if find_val_tch:
            loss_val_tch.append(float(find_val_tch.group(1)))
            acc_val_tch.append(float(find_val_tch.group(2)))

# print(len(loss_train))
# print(len(loss_val))
# print(len(loss_val_tch))

fig, ax = plt.subplots(1,2, figsize=(15,6))
ax[0].plot(range(1, len(loss_train)+1), loss_train, label='training')
ax[0].plot(range(1, len(loss_val)+1), loss_val, label='validation')
if len(loss_val_tch) > 0:
    ax[0].plot(range(6, len(loss_val_tch)+1), loss_val_tch[5:], label='tch validation') # skip the first 5 epochs
ax[0].set_xticks(np.arange(0, len(loss_train)+1, 10))
ax[0].set_xlabel('Epochs', fontsize=16)
ax[0].set_ylabel('Loss', fontsize=16)
ax[0].grid()
ax[0].legend(fontsize=16)
ax[1].plot(range(1, len(acc_train)+1), acc_train, label='training')
ax[1].plot(range(1, len(acc_val)+1), acc_val, label='validation')
if len(acc_val_tch) > 0:
    ax[1].plot(range(6, len(acc_val_tch)+1), acc_val_tch[5:], label='tch validation') # skip the first 5 epochs
ax[1].set_xticks(np.arange(0, len(acc_train)+1, 10))
ax[1].set_yticks(np.arange(0, 100, 10))
ax[1].set_xlabel('Epochs', fontsize=16)
ax[1].set_ylabel('Accuracy (\%)', fontsize=16)
ax[1].grid()
ax[1].legend(fontsize=16)
plt.suptitle('Learning Curve', fontsize=20)
fig.savefig(learning_curve_fname, bbox_inches='tight')
plt.close(fig)
