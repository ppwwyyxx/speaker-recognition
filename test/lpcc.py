#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: lpcc.py
# Date: Wed Dec 25 14:10:49 2013 +0000
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import time
import scikits.talkbox as tb
from scikits.talkbox.linpred import levinson_lpc
import numpy as np
from numpy import *
from scipy.io import  wavfile

def hamming(n):
    """ Generate a hamming window of n points as a numpy array.  """
    return 0.54 - 0.46 * cos(2 * pi / n * (arange(n) + 0.5))

class LPCCExtractor(object):
    def __init__(self, fs, n_lpc=10, n_lpcc=13):
        self.PRE_EMPH = 0.95
        self.n_lpc = n_lpc
        self.n_lpcc = n_lpcc + 1

        self.FRAME_LEN = int(0.02 * fs)
        self.FRAME_SHIFT = int(0.01 * fs)
        self.window = hamming(self.FRAME_LEN)


    def lpc_to_cc(self, lpc):
        lpcc = np.zeros(self.n_lpcc)
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
        lpcc = self.lpc_to_cc(lpc)
        return lpcc

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


def main():
    extractor = LPCCExtractor(8000)
    fs, signal = wavfile.read("../corpus.silence-removed/Style_Reading/f_001_03.wav")
    start = time.time()
    ret = extractor.extract(signal)
    print len(ret)
    print len(ret[0])
    print time.time() - start


if __name__ == "__main__":
   main()
