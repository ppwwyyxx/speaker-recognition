#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: test.py
# $Date: Mon Dec 16 04:27:02 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import pygmm
from sklearn.mixture import GMM as SKGMM
from numpy import *
import numpy as np
import sys


def read_data(fname):
    with open(fname) as fin:
        return map(lambda line: map(float, line.rstrip().split()), fin)

def get_gmm(where):
    nr_mixture = 256
    nr_iteration = 1
    if where == 'pygmm':
        return pygmm.GMM(nr_mixture = nr_mixture,
                min_covar = 1e-3,
                nr_iteration = nr_iteration,
                init_with_kmeans = 0,
                concurrency = 1)
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
#X.extend(X + X + X)
#X = random_vector(100, 13)
#extend_X(X, 10)
#print(len(X))


#gmm_type = 'sklearn'

def test():
    for nr_instance in [1000, 2000, 4000, 8000, 16000, 32000, 64000,
        128000, 256000, 512000]:
        for gmm_type in ['pygmm', 'sklearn']:
            print(gmm_type)
            gmm = get_gmm(gmm_type)
            import time
            start = time.time()
            gmm.fit(X[:nr_instance])
            print("result {} {} : {}" . format(
                gmm_type, nr_instance, time.time() - start))
            sys.stdout.flush()

test()

# vim: foldmethod=marker

