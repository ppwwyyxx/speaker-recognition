#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: test-mixture.py
# $Date: Thu Dec 26 02:07:32 2013 +0000
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import glob
import traceback
import sys
import random
import os
import time
import numpy as np
import multiprocessing
from multiprocess import MultiProcessWorker
import operator
from collections import defaultdict
from sklearn.mixture import GMM

concurrency = multiprocessing.cpu_count()
from feature import BOB, LPC, MFCC, get_extractor
from sample import Sample

class Person(object):
    def __init__(self, sample = None, name = None, gender = None):
        self.sample = sample
        self.name = name
        self.gender = gender
        self.samples = []

    def add_sample(self, sample):
        self.samples.append(sample)
        if not self.sample:
            self.sample = sample
        else:
            self.sample.add(sample)

    def sample_duration(self):
        return self.sample.duration()

    def get_fragment(self, duration):
        return self.sample.get_fragment(duration)

    def get_fragment_with_interval(self, duration):
        return self.sample.get_fragment_with_interval(duration)

    def remove_subsignal(self, begin, end):
        self.sample.remove_subsignal(begin, end)


def get_corpus(dirs):
    persons = defaultdict(Person)

    for d in dirs:
        print("processing {} ..." . format(d))
        for fname in sorted(glob.glob(os.path.join(d, "*.wav"))):
            basename = os.path.basename(fname)
            gender, name, _ = basename.split('_')
            p = persons[name]
            p.name, p.gender = name, gender
            try:
                orig_sample = Sample.from_wavfile(fname)
                p.add_sample(orig_sample)
            except Exception as e:
                print("Exception occured while reading {}: {} " . format(
                    fname, e))
                print("======= traceback =======")
                print(traceback.format_exc())
                print("=========================")

    return persons

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

def gen_data(params):
    p, duration = params
    return p.get_fragment(duration)

def predict_task(gmmset, x_test):
    return gmmset.predict_one(x_test)

def test_feature(feature_impl, X_train, y_train, X_test, y_test):
    start = time.time()
    print 'calculating features...',
    worker = MultiProcessWorker(feature_impl)
    X_train = worker.run(X_train)
    del worker
    worker = MultiProcessWorker(feature_impl)
    X_test = worker.run(X_test)
    del worker
    print 'time elapsed: ', time.time() - start

    for mixture in [16, 32, 48, 64, 80, 96, 120, 144]:
        start = time.time()
        gmmset = GMMSet(mixture)
        print "nmixture: ", mixture
        print 'training ...',
        gmmset.fit(X_train, y_train)
        nr_correct = 0
        print 'time elapsed: ', time.time() - start

        print 'predicting...',
        start = time.time()
        pool = multiprocessing.Pool(concurrency)
        predictions = []
        for x_test, label_true in zip(*(X_test, y_test)):
            predictions.append(pool.apply_async(predict_task, args = (gmmset, x_test)))
        pool.close()
        for ind, (x_test, label_true) in enumerate(zip(*(X_test, y_test))):
            label_pred = predictions[ind].get()
            if label_pred == label_true:
                nr_correct += 1
        print 'time elapsed: ', time.time() - start
        print("{}/{} {:.6f}".format(nr_correct, len(y_test),
                float(nr_correct) / len(y_test)))

def main():
    if len(sys.argv) == 1:
        print("Usage: {} <dir_contains_wav_file> [<dirs> ...]" . format(
                sys.argv[0]))
        sys.exit(1)

    dirs = sys.argv[1:]

    nr_person = 50
    train_duration = 15
    test_duration = 5
    nr_test_fragment_per_person = 50

    persons = list(get_corpus(dirs).iteritems())
    random.shuffle(persons)
    persons = persons[:nr_person]

    print('generating data ...')
    X_train, y_train = [], []
    X_test, y_test = [], []
    for name, p in persons:
        y_train.append(name)
        fs, signal, begin, end = p.get_fragment_with_interval(train_duration)
        # it is important to remove signal used for training to get
        # unbiased result
        p.remove_subsignal(begin, end)
        X_train.append((fs, signal))
    for name, p in persons:
        for i in xrange(nr_test_fragment_per_person):
            y_test.append(name)
            X_test.append(gen_data((p, test_duration)))

    def mix(tup):
        bob = BOB.extract(tup)
        lpc = LPC.extract(tup)
        return np.concatenate((bob, lpc), axis=1)

    fout = open("final-log/nmixture.log", 'a')
    sys.stdout = fout
    test_feature(mix, X_train, y_train, X_test, y_test)
    test_feature(mix, X_train, y_train, X_test, y_test)
    test_feature(mix, X_train, y_train, X_test, y_test)
    test_feature(mix, X_train, y_train, X_test, y_test)
    fout.close()

if __name__ == '__main__':
    main()


# vim: foldmethod=marker

