#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: test.py
# $Date: Wed Dec 11 18:56:21 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import pygmm
from sklearn.mixture import GMM as SKGMM
from numpy import *
import numpy as np


def read_data(fname):
    with open(fname) as fin:
        return map(lambda line: map(float, line.rstrip().split()), fin)

def get_gmm(where):
    nr_mixture = 256
    nr_iteration = 1000
    if where == 'pygmm':
        return pygmm.GMM(nr_mixture = nr_mixture,
                min_covar = 1e-3,
                nr_iteration = nr_iteration,
                concurrency = 4)
    elif where == 'sklearn':
        return SKGMM(nr_mixture, n_iter = nr_iteration)
    return None

def random_vector(n, dim):
    import random
    ret = []
    for j in range(n):
        ret.append([random.random() for i in range(dim)])
    return ret

def extend_X(X, n):
    import copy
    Xt = copy.deepcopy(X)
    for i in range(n - 1):
        X.extend(Xt)

X = read_data('../test.data')
X.extend(X + X + X)
#X = random_vector(100, 13)
#extend_X(X, 10)
#print(len(X))


#gmm_type = 'sklearn'

global gmm

def timing(code):
    global gmm
    import time
    start = time.time()
    exec(code)
    print(time.time() - start)

def test():
    global gmm
    for gmm_type in ['pygmm', 'sklearn']:
        print(gmm_type)
        gmm = get_gmm(gmm_type)
        timing("gmm.fit(X)")
        if gmm_type == 'pygmm':
            gmm.dump('gmm.model')
        print(np.sum(gmm.score(X)))
        print("-------")

test()

# vim: foldmethod=marker

