#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: datautil.py
# $Date: Wed Dec 25 01:10:24 2013 +0000
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import os
import glob
import random
import config

import numpy as np
from numpy import loadtxt

def load_train_test(fname, training_data_len, nr_test, testcase_len):
    """
    :return: training_data, test_data"""
    data =loadtxt(fname)
    data_len = len(data)
    training_data = data[:training_data_len]
    test_data = []
    for i in xrange(nr_test):
        start_pos = random.randint(training_data_len, data_len - testcase_len)
        end_pos = start_pos + testcase_len
        test_data.append(data[start_pos:end_pos])

    test_data = np.array(test_data)
    return training_data, test_data

def main():
    fname = 'test-data/mfcc-data/Style_Reading/f_001_03.mfcc'
    frame_duration = 0.02
    frame_shift = 0.01
    nr_frame_per_sec = int(1.0 / frame_shift)
    fs = 8000
    train_duration = 30.0
    train_len = int(train_duration * nr_frame_per_sec)
    nr_test = 100
    test_duration = 5.0
    testcase_len = int(test_duration * nr_frame_per_sec)

    train, test = load_train_test(fname, train_len, nr_test, testcase_len)

def read_raw_data(pattern):
    """:return X"""
    if isinstance(pattern, basestring):
        fpaths = glob.glob(pattern)
    elif isinstance(pattern, list):
        fpaths = pattern

    X = []
    for fpath in fpaths:
        print 'loading file {} ... ' . format(fpath)
        X.extend(loadtxt(fpath))
    return X


def read_data(pattern, nr_person, shuffle=True):
    """
    :param pattern, a file name pattern or file list
    :return X_train, y_train, X_test, y_test"""
    X_train, y_train, X_test, y_test = [], [], [], []
    if isinstance(pattern, basestring):
        fpaths = sorted(glob.glob(pattern))
    elif isinstance(pattern, list):
        fpaths = pattern
    if shuffle:
        random.shuffle(fpaths)
    for fpath in fpaths[:nr_person]:
        print 'reading {} ...' . format(fpath)
        fname = os.path.basename(fpath)
        base, ext = os.path.splitext(fname)
        x_train, x_test = load_train_test(fpath,
                config.train_len,
                config.nr_test,
                config.testcase_len)
        X_train.append(x_train)
        X_test.extend(x_test)
        y_train.append(base)
        y_test.extend([base] * len(x_test))

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    main()

# vim: foldmethod=marker
