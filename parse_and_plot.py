# usage: python3 parse_and_plot.py checkpoint/wcl_asym04_run0_M4n10rho05w1_mLR05_ep150_log

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, re, os
import numpy as np

loss_tch = []
loss_stu = []
acc_tch = []
acc_stu = []
ite_tch = []
ite_stu = []
with open(sys.argv[1], 'r') as fhand:
    for line in fhand:
        line = line.strip()
        if re.search('tch Validation Epoch', line):
            search_results = re.search('Epoch \#([0-9| ]+) Batch \#([0-9| ]+).*Loss: ([0-9]+\.[0-9]+) Acc@1: ([0-9]+\.[0-9]+)\%', line)
            ite_tch.append((int(search_results.group(1)) - 1) * 352 + int(search_results.group(2)))
            loss_tch.append(float(search_results.group(3)))
            acc_tch.append(float(search_results.group(4)))
        elif re.search('Validation Epoch', line):
            search_results = re.search('Epoch \#([0-9| ]+) Batch \#([0-9| ]+).*Loss: ([0-9]+\.[0-9]+) Acc@1: ([0-9]+\.[0-9]+)\%', line)
            ite_stu.append((int(search_results.group(1)) - 1) * 352 + int(search_results.group(2)))
            loss_stu.append(float(search_results.group(3)))
            acc_stu.append(float(search_results.group(4)))

fig, ax = plt.subplots(1, 2, figsize=(15,6))
ax[0].plot(ite_tch, loss_tch, label='tch')
ax[0].plot(ite_stu, loss_stu, label='stu')
#ax[0].set_xticks(ite_tch)
ax[0].set_xlabel('Training iteration', fontsize=16)
ax[0].set_ylabel('Validation loss', fontsize=16)
ax[0].set_ylim(0, 2)
ax[0].legend(fontsize=16)
ax[1].plot(ite_tch, acc_tch, label='tch')
ax[1].plot(ite_stu, acc_stu, label='stu')
#ax[1].set_xticks(ite_tch)
ax[1].set_xlabel('Training iteration', fontsize=16)
ax[1].set_ylabel('Validation accuracy', fontsize=16)
ax[1].set_ylim(60, 100)
ax[1].legend(fontsize=16)
plt.suptitle('Learning Curve', fontsize=20)
fig.savefig(sys.argv[1].replace('_log', '_plot.png'),
            bbox_inches='tight')
plt.close(fig)


