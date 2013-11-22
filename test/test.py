#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test.py
# Date: Mon Nov 18 11:43:45 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import operator
import glob
from random import choice

import scipy.io.wavfile as wavfile
import numpy as np
from sklearn.mixture import GMM

from MFCC import get_mfcc_extractor

dirs = ['../test-data/xinyu',
        '../test-data/twb',
        '../test-data/ly',
        '../test-data/zdw',
        '../test-data/qq',
#        '../test-data/fsew0_v1.1/converted/',
#        '../test-data/maps0/converted/',
#        '../test-data/msak0_v1.1/converted/'
        ]

mfccs = []
print "reading and calculating..."


for d in dirs:
    features = []
    print d
    for i in range(3):
        f = choice(glob.glob(d + "/*.wav"))
        fs, signal = wavfile.read(f)
        extractor = get_mfcc_extractor(fs)
        mfcc = extractor.extract(signal, True)
        features.extend(mfcc)
    mfccs.append(features)

def gmm_model_feature(model):
    return np.append(model.means_, [model.weights_])

print "start training"
gmms = []
for idx, mfcc in enumerate(mfccs):
    print idx
    gmm = GMM(32, n_iter=1000, thresh=0.001)
    gmm.fit(mfcc)
    print gmm_model_feature(gmm)
    gmms.append(gmm)

print "done training"

def cal_score(model, mfcc):
    return np.exp(sum(model.score(mfcc)) / 1000)

def pred_label(mfcc):
    scores = [cal_score(gmm, mfcc) for gmm in gmms]
    return max(enumerate(scores), key=operator.itemgetter(1))[0]

cnt = 0
right = 0
for idx, d in enumerate(dirs):
    for f in glob.glob(d + "/*.wav"):
        cnt += 1
        fs, signal = wavfile.read(f)
        mfcc = get_mfcc_extractor(fs).extract(signal, True)
        pred = pred_label(mfcc)
        if idx == pred:
            right += 1
        print f, idx, pred

print "Count: ", cnt, right
print "Error rate: ", float(cnt - right) / cnt
