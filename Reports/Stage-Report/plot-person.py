#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: plot-person.py
# Date: Mon Dec 16 04:00:20 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import defaultdict


def get_data(fname):

    data = defaultdict(list)
    with open(fname) as f:
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

    x = data2[:,0]
    y = data2[:,1]

    yerr = data2[:,2]
    return x, y, yerr


plt.figure()
plt.grid(color = 'gray', linestyle = 'dashed')
plt.ylim([0.85, 1])

x, y, yerr = get_data('person_log/all')
plt.plot(x, y, lw=2, label='GMM from scikit-learn', color='#9061c2')

x, y, yerr = get_data('person_log_newgmm/all')
plt.plot(x, y, lw=2, label='Our GMM', color='green')
plt.legend(fancybox=True, shadow=True, loc=4)
plt.savefig('out.pdf')


plt.show()
