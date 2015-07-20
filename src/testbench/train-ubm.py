#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: train-ubm.py
# $Date: Fri Dec 27 04:19:58 2013 +0000
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import multiprocessing

import glob
import config
import random
import datautil

from gmmset import GMM
#from sklearn.mixture import GMM

nr_mixture = 32

def get_gmm():
    from sklearn.mixture import GMM as skGMM
    from gmmset import GMM as pyGMM
    if GMM == skGMM:
        print 'using GMM from sklearn'
        return GMM(nr_mixture)
    else:
        print 'using pyGMM'
        return GMM(nr_mixture=nr_mixture, nr_iteration=500,
                init_with_kmeans=0, concurrency=8,
                threshold=1e-15,
                verbosity=2)


def get_all_data_fpaths():
    fpaths = []
    for style in ['Style_Reading', 'Style_Spontaneous', 'Style_Whisper']:
        fpaths.extend(glob.glob('test-data/mfcc-lpc-data/{}/*.mfcc-lpc' .
            format(style)))
    return fpaths


def train_all_together_ubm():
    global nr_mixture
    nr_utt_in_ubm = 300
    fpaths = get_all_data_fpaths()
    random.shuffle(fpaths)
    fpaths = fpaths[:nr_utt_in_ubm]
    X = datautil.read_raw_data(fpaths)
    gmm = get_gmm()
    gmm.fit(X)
    gmm.dump('model/ubm.mixture-{}.utt-{}.model' . format(
        nr_mixture, nr_utt_in_ubm))

def main():
    train_all_together_ubm()
    return


if __name__ == '__main__':
    main()

# vim: foldmethod=marker
