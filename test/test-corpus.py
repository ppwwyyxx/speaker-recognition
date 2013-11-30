#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: test-corpus.py
# $Date: Sat Nov 30 18:52:56 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import glob
import traceback
import sys
import random
import os
import scipy.io.wavfile as wavfile
import numpy as np
import multiprocessing
import operator
from collections import defaultdict
from sklearn.mixture import GMM

import MFCC

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

#    dirs = [ '../test-data/corpus.silence-removed//Style_Reading',
#            '../test-data/corpus.silence-removed//Style_Spontaneous',
#            '../test-data/corpus.silence-removed//Style_Whisper',
#            ]
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
    return MFCC.extract(*p.get_fragment(duration))

def predict_task(gmmset, x_test):
    return gmmset.predict_one(x_test)

def main():

    if len(sys.argv) == 1:
        print("Usage: {} <dir_contains_wav_file> [<dirs> ...]" . format(
                sys.argv[0]))
        sys.exit(1)

    dirs = sys.argv[1:]

    nr_person = 20
    train_duration = 30
    test_duration = 5
    nr_test_fragment_per_person = 100
    concurrency = 4

    persons = list(get_corpus(dirs).iteritems())
    random.shuffle(persons)

    persons = dict(persons[:nr_person])

    print('generating data ...')
    X_train, y_train = [], []
    X_test, y_test = [], []
    for name, p in persons.iteritems():
        print(name, p.sample_duration())
        y_train.append(name)
        fs, signal, begin, end = p.get_fragment_with_interval(train_duration)
        # it is important to remove signal used for training to get
        # unbiased result
        p.remove_subsignal(begin, end)

        X_train.append(MFCC.extract(fs, signal))

        # return x

        pool = multiprocessing.Pool(concurrency)
        params = []
        for i in xrange(nr_test_fragment_per_person):
            params.append((p, test_duration))
            y_test.append(name)
#            X_test.append(MFCC.extract(*p.get_fragment(test_duration)))
#            y_test.append(name)
        X_test.extend(pool.map(gen_data, params))

    gmmset = GMMSet()
    print('training ...')
    gmmset.fit(X_train, y_train)
    nr_correct = 0

    pool = multiprocessing.Pool(concurrency)
    predictions = []
    for x_test, label_true in zip(*(X_test, y_test)):
        predictions.append(pool.apply_async(predict_task, args = (gmmset, x_test)))
    pool.close()
    for ind, (x_test, label_true) in enumerate(zip(*(X_test, y_test))):
        label_pred = predictions[ind].get()
        is_wrong = '' if label_pred == label_true else ' wrong'
        print("{} {}{}" . format(label_pred, label_true, is_wrong))
        if label_pred == label_true:
            nr_correct += 1
    print("{}/{} {:.2f}" . format(
            nr_correct, len(y_test),
            float(nr_correct) / len(y_test)))

    print(dirs)
    print(nr_person, train_duration, test_duration, nr_test_fragment_per_person)

if __name__ == '__main__':
    main()


# vim: foldmethod=marker

