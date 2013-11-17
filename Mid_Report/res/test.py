#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test.py
# Date: Sun Nov 17 22:20:12 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from MFCC import MFCCExtractor
import operator
from random import choice

import scipy.io.wavfile as wavfile
import numpy as np
import glob

from sklearn.mixture import GMM

dirs = ['data1', 'data2', 'data3']
nspeaker = len(dirs)

mfccs = []
print "reading and calculating..."

extractor = MFCCExtractor(16000)

for d in dirs:
    features = []
    print d
    for i in range(2):
        f = choice(glob.glob(d + "/*.wav"))
        fs, signal = wavfile.read(f)
        mfcc = extractor.extract_differential(signal)
        features.extend(mfcc)
    mfccs.append(features)

print "start training"
gmms = []
for idx, mfcc in enumerate(mfccs):
    print idx
    gmm = GMM(32, n_iter=1000, thresh=0.001)
    gmm.fit(mfcc)
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
        mfcc = extractor.extract_differential(signal)
        pred = pred_label(mfcc)
        print f, idx, pred
        if idx == pred:
            right += 1

print "Count: ", cnt, right
print "Accuracy: ", float(right) / cnt
