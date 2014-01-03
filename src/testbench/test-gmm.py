#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: test-gmm.py
# $Date: Fri Dec 27 01:42:37 2013 +0000
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import os
import glob

import config
import datautil

from gmm.python.pygmm import GMM
#from gmmset import GMMSet
from gmmset import GMMSetPyGMM as GMMSet

def get_training_data_fpaths():
    fpaths = []
    for fpath in sorted(glob.glob('test-data/mfcc-data/Style_Reading/*.mfcc')):
        fname = os.path.basename(fpath)
        base, ext = os.path.splitext(fname)
        if base in config.ubm_set:
            continue
        fpaths.append(fpath)
    return fpaths


def load_gmmset(labels, nr_person):
    gmmset = GMMSet(concurrency=8)
    for fpath in sorted(glob.glob('model.new-mfcc/*')):
        fname = os.path.basename(fpath)
        base = fname[:fname.find('.')]
        if base not in labels:
            continue
        if fname.endswith("32.model"):
            print base, fname
            gmmset.load_gmm(base, fpath)
    return gmmset

def main():
    nr_person = 20
    fpaths = get_training_data_fpaths()
    X_train, y_train, X_test, y_test = datautil.read_data(
            fpaths, nr_person)

    print "loading gmms ..."
    gmmset = load_gmmset(y_train, nr_person)

#    print "training ..."
#    ubm = GMM.load(config.ubm_model_file)
#    ubm = None
#    gmmset = GMMSet(32,ubm=ubm, concurrency=8,
#            verbosity=1, nr_iteration=100,
#            threshold=1e-2)
#    gmmset.fit(X_train, y_train)

    print "predicting ..."
    import time
    start = time.time()
    import cProfile
    y_pred = gmmset.predict(X_test)
    print time.time() - start

    nr_total = len(y_test)
    nr_correct = len(filter(lambda x: x[0] == x[1], zip(y_pred, y_test)))
    print "{} {}/{}" . format(
        float(nr_correct) / nr_total, nr_correct, nr_total)
    print "nr_person: {}" . format(nr_person)

if __name__ == '__main__':
    main()
#    import cProfile
#    cProfile.run("main()")

# vim: foldmethod=marker
