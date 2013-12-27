#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: plot.py
# Date: Mon Dec 16 02:46:51 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import defaultdict



data = defaultdict(list)

with open('all') as f:
    for line in f:
        line = line.strip().split(':')[1]
        line = line.split()
        nperson = int(line[0].split('/')[1]) / 100
        if nperson == 2:
            continue
        score = float(line[1])
        data[nperson].append(score)

data2 = []
for nperson, scorelist in data.iteritems():
    avg = np.average(scorelist)
    std = np.std(scorelist)
    data2.append((nperson, avg, std))

data2.sort()



data2 = np.array(data2)

# example data
x = data2[:,0]
y = data2[:,1]

# example variable error bar values
yerr = data2[:,2]

# First illustrate basic pyplot interface, using defaults where possible.
plt.figure()

plt.grid(color = 'gray', linestyle = 'dashed')
plt.ylim([0.85, 1])
plt.plot(x, y, lw=2)


plt.show()
