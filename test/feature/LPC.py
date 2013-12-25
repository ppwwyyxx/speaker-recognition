#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: LPC.py
# Date: Wed Dec 25 15:53:51 2013 +0000
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import time
#import scikits.talkbox as tb
from scikits.talkbox.linpred import levinson_lpc
from numpy import *
from scipy.io import  wavfile
from MFCC import hamming
from utils import cached_func

class LPCExtractor(object):
    def __init__(self, fs, win_length_ms, win_shift_ms, n_lpc,
                 pre_emphasis_coef):
        self.PRE_EMPH = pre_emphasis_coef
        self.n_lpc = n_lpc
        #self.n_lpcc = n_lpcc + 1

        self.FRAME_LEN = int(float(win_length_ms) / 1000 * fs)
        self.FRAME_SHIFT = int(float(win_shift_ms) / 1000 * fs)
        self.window = hamming(self.FRAME_LEN)


    def lpc_to_cc(self, lpc):
        lpcc = zeros(self.n_lpcc)
        lpcc[0] = lpc[0]
        for n in range(1, self.n_lpc):
            lpcc[n] = lpc[n]
            for l in range(0, n):
                lpcc[n] += lpc[l] * lpcc[n - l - 1] * (n - l) / (n + 1)
        for n in range(self.n_lpc, self.n_lpcc):
            lpcc[n] = 0
            for l in range(0, self.n_lpc):
                lpcc[n] += lpc[l] * lpcc[n - l - 1] * (n - l) / (n + 1)
        return -lpcc[1:]

    def lpcc(self, signal):
        lpc = levinson_lpc.lpc(signal, self.n_lpc)[0]
        return lpc[1:]
        #lpcc = self.lpc_to_cc(lpc)
        #return lpcc

    def extract(self, signal, diff=False):
        frames = (len(signal) - self.FRAME_LEN) / self.FRAME_SHIFT + 1
        feature = []
        for f in xrange(frames):
            frame = signal[f * self.FRAME_SHIFT : f * self.FRAME_SHIFT +
                           self.FRAME_LEN] * self.window
            frame[1:] -= frame[:-1] * self.PRE_EMPH
            feature.append(self.lpcc(frame))

        if diff:
            ret = []
            for feat in feature:
                diff = lambda f: [x - f[i - 1] for i, x in enumerate(f)][1:]
                diff1 = diff(feat)
                #diff2 = diff(diff1)
                ret.append(concatenate((feat, diff1)))
            return ret
        return feature

@cached_func
def get_lpc_extractor(fs, win_length_ms=32, win_shift_ms=16,
                       n_lpc=15, pre_emphasis_coef=0.95):
    ret = LPCExtractor(fs, win_length_ms, win_shift_ms, n_lpc, pre_emphasis_coef)
    return ret


def extract(fs, signal=None, **kwargs):
    """accept two argument, or one as a tuple"""
    if signal is None:
        assert type(fs) == tuple
        fs, signal = fs[0], fs[1]
    signal = cast['float'](signal)
    return get_lpc_extractor(fs, **kwargs).extract(signal)

if __name__ == "__main__":
    extractor = LPCCExtractor(8000)
    fs, signal = wavfile.read("../corpus.silence-removed/Style_Reading/f_001_03.wav")
    start = time.time()
    ret = extractor.extract(signal)
    print len(ret)
    print len(ret[0])
    print time.time() - start
