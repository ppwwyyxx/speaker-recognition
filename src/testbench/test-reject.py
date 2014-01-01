#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: test-reject.py
# $Date: Fri Dec 27 03:24:35 2013 +0000
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import datautil
from gmmset import GMM
from gmmset import GMMSetPyGMM as GMMSet
from feature import BOB, LPC
from scipy.io import wavfile
import random
import numpy as np

def mix_feature(tup):
    bob = BOB.extract(tup)
    lpc = LPC.extract(tup)
    return np.concatenate((bob, lpc), axis=1)

def monotize_signal(signal):
    if signal.ndim > 1:
        signal = signal[:,0]
    return signal

def test_ubm_var_channel():
    ubm = GMM.load('model/ubm.mixture-32.person-20.immature.model')

    train_duration = 8.
    nr_test = 5
    test_duration = 3.
    audio_files = ['xinyu.vad.wav', 'wyx.wav']
    X_train, y_train, X_test, y_test = [], [], [], []
    for audio_file in audio_files:
        fs, signal = wavfile.read(audio_file)
        signal = monotize_signal(signal)

        train_len = int(fs * train_duration)
        test_len = int(fs * test_duration)

        X_train.append(mix_feature((fs, signal[:train_len])))
        y_train.append(audio_file)

        for i in range(nr_test):
            start = random.randint(train_len, len(signal) - test_len)
            X_test.append(mix_feature((fs, signal[start:start+train_len])))
            y_test.append(audio_file)

    gmmset = GMMSet(32, ubm=ubm)
    gmmset.fit(X_train, y_train)
    y_pred = gmmset.predict_with_reject(X_test)
    for i in xrange(len(y_pred)):
        print y_test[i], y_pred[i], '' if y_test[i] == y_pred[i] else 'wrong'

    for imposter_audio_file in map(
            lambda x: 'test-{}.wav'.format(x), range(5)):
        fs, signal = wavfile.read(imposter_audio_file)
        signal = monotize_signal(signal)
        imposter_x = mix_feature((fs, signal))
        print gmmset.predict_one_with_rejection(imposter_x)

test_ubm_var_channel()
import sys
sys.exit(0)

ubm = GMM.load('model/ubm.mixture-32.person-20.immature.model')
gmm = GMM(32, verbosity=1)

#audio_file = 'test-data/corpus.silence-removed/Style_Reading/f_001_03.wav'

fs, signal = wavfile.read(audio_file)
signal = monotize_signal(signal)
X = mix_feature((fs, signal))

ubm = GMM.load('model/ubm.mixture-32.person-20.immature.model')
gmm = GMM(32, verbosity=1)

X = X[:1000]
gmm.fit(X, ubm=ubm)
gmm.dump('xinyu.model')

# vim: foldmethod=marker

