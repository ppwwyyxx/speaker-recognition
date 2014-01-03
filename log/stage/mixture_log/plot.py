#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: plot.py
# Date: Mon Dec 16 03:26:04 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import defaultdict
import sys



data = defaultdict(list)

with open(sys.argv[1]) as f:
    cnt = 0
    for line in f:
        print line
        line = line.split()
        score = float(line[1])
        nmixture = 2 ** (cnt + 1)
        cnt = (cnt + 1) % 6
        data[nmixture].append(score)
    print cnt

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
plt.ylim([0.60, 1])
plt.plot(x, y, lw=2)


plt.show()
