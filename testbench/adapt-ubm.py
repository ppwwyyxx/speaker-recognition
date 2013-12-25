#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: adapt-ubm.py
# $Date: Wed Dec 25 13:24:17 2013 +0000
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import os
import glob

from gmmset import GMM
import config
import datautil


def get_training_data_fpaths():
    fpaths = []
    for fpath in glob.glob('test-data/mfcc-data/Style_Reading/*.mfcc'):
        fname = os.path.basename(fpath)
        base, ext = os.path.splitext(fname)
        if base in config.ubm_set:
            continue
        fpaths.append(fpath)
    return fpaths

def main():
    nr_person = 50
    fpaths = get_training_data_fpaths()
    X_train, y_train, X_test, y_test = datautil.read_data(fpaths, nr_person)
    ubm = GMM.load('model/ubm-1024.model')
    for x, y in zip(X_train, y_train):
        gmm = GMM(concurrency=8,
                threshold=0.0001,
                nr_iteration=400,
                verbosity=1)
        gmm.fit(x, ubm=ubm)
        gmm.dump("model/" + y + ".1024.model")

if __name__ == '__main__':
    main()

# vim: foldmethod=marker
