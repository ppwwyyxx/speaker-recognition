#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: train-ubm.py
# $Date: Wed Dec 25 01:36:59 2013 +0000
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import multiprocessing

import config
import datautil

from gmmset import GMM

def main():
#    file_pattern = 'test-data/mfcc-data/Style_Reading/*.mfcc'
#    X_train, y_train, X_test, y_test = datautil.read_data(file_pattern, 10)
    fpaths = map(lambda x: 'test-data/mfcc-data/Style_Reading/' + x + '.mfcc',
            config.ubm_set)
    X = datautil.read_raw_data(fpaths)
    gmm = GMM(nr_mixture=1024, nr_iteration=500,
            init_with_kmeans=0, concurrency=8,
            verbosity=2)
    gmm.fit(X)
    gmm.dump('model/ubm-1024.model')


if __name__ == '__main__':
    main()

# vim: foldmethod=marker
