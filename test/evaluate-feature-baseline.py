#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: test-feature.py
# $Date: Wed Dec 11 22:01:13 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import glob
import traceback
import sys
import random
import os
import time
import numpy as np
from itertools import izip
import multiprocessing
import operator
from collections import defaultdict
from sklearn.mixture import GMM

concurrency = multiprocessing.cpu_count()

import BOB as bob_MFCC
import MFCC

from sample import Sample

class GMMSet(object):
    def __init__(self, gmm_order = 32):
        self.gmms = []
        self.gmm_order = 32
        self.y = []

    def fit_new(self, x, label):
        self.y.append(label)
        gmm = GMM(self.gmm_order)
        gmm.fit(x)
        self.gmms.append(gmm)

    def cluster_by_label(self, X, y):
        Xtmp = defaultdict(list)
        for ind, x in enumerate(X):
            label = y[ind]
            Xtmp[label].extend(x)
        yp, Xp = zip(*Xtmp.iteritems())
        return Xp, yp

    def fit(self, X, y):
        X, y = self.cluster_by_label(X, y)
        for ind, x in enumerate(X):
            self.fit_new(x, y[ind])

    def gmm_score(self, gmm, x):
        return np.exp(np.sum(gmm.score(x)) / 1000)
    def predict_one(self, x):
        scores = [self.gmm_score(gmm, x) for gmm in self.gmms]
        return self.y[max(enumerate(scores), key = operator.itemgetter(1))[0]]

    def predict(self, X):
        return map(self.predict_one, X)

def predict_task(gmmset, x_test):
    return gmmset.predict_one(x_test)

def do_test(X_train, y_train, X_test, y_test):
    start = time.time()
    gmmset = GMMSet()
    print('training ...')
    gmmset.fit(X_train, y_train)
    nr_correct = 0
    print 'time elapsed: ', time.time() - start

    print 'predicting...'
    start = time.time()
    pool = multiprocessing.Pool(concurrency)
    predictions = []
    for x_test, label_true in izip(X_test, y_test):
        predictions.append(pool.apply_async(predict_task, args = (gmmset, x_test)))
    for ind, (x_test, label_true) in enumerate(zip(X_test, y_test)):
        label_pred = predictions[ind].get()
        is_wrong = '' if label_pred == label_true else ' wrong'
        print("{} {}{}" . format(label_pred, label_true, is_wrong))
        if label_pred == label_true:
            nr_correct += 1
    print 'time elapsed: ', time.time() - start
    print("{}/{} {:.2f}".format(nr_correct, len(y_test),
            float(nr_correct) / len(y_test)))
    pool.close()

def main():
    if len(sys.argv) == 1:
        print("Usage: {} <dir_contains_feature_file>" . format(
                sys.argv[0]))
        sys.exit(1)

    dirs = sys.argv[1]

    print('reading data ...')
    X_train, y_train = [], []
    with open(os.path.join(dirs, 'enroll.lst')) as f:
        for line in f:
            line = line.split('=')
            label = int(line[0])
            fname = line[1].strip().rsplit('/')[1]
            #print label, fname
            mat = []
            with open(os.path.join(dirs, fname)) as feaf:
                for line in feaf:
                    line = map(float, line.strip().split())
                    line = np.array(line)
                    mat.append(line)
            X_train.append(np.array(mat))
            y_train.append(label)
    print "length of X_train: ", len(X_train)

    X_test, y_test = [], []
    with open(os.path.join(dirs, 'test.lst')) as f:
        for line in f:
            line = line.split('=')
            label = int(line[0])
            fname = line[1].strip().rsplit('/')[1]
            mat = []
            with open(os.path.join(dirs, fname)) as feaf:
                for line in feaf:
                    line = map(float, line.strip().split())
                    mat.append(np.array(line))
            X_test.append(np.array(mat))
            y_test.append(label)
    do_test(X_train, y_train, X_test, y_test)

    print(dirs)

if __name__ == '__main__':
    main()


# vim: foldmethod=marker

