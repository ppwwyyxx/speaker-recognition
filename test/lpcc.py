#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: lpcc.py
# Date: Tue Dec 24 20:40:24 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import time
import scikits.talkbox as tb
import numpy as np
from scipy.io import  wavfile

class LPCCExtractor(object):
    def __init__(self, fs, n_lpc=10, n_lpcc=13):
        self.n_lpc = n_lpc
        self.n_lpcc = n_lpcc + 1

        self.FRAME_LEN = int(0.02 * fs)
        self.FRAME_SHIFT = int(0.01 * fs)

    def lpc_to_cc(self, lpc):
        lpcc = np.zeros(self.n_lpcc)
        for n in range(1, self.n_lpc):
            lpcc[n] = lpc[n]
            for l in range(1, n):
                lpcc[n] += lpc[l] * lpcc[n - l] * (n - l) / n
        for n in range(self.n_lpc, self.n_lpcc):
            lpcc[n] = 0
            for l in range(1, self.n_lpc + 1):
                lpcc[n] += lpc[l] * lpcc[n - l] * (n - l) / n
        return -lpcc[1:]

    def lpcc(self, signal):
        lpc = tb.lpc(signal, self.n_lpc)[0]
        lpcc = self.lpc_to_cc(lpc)
        return lpcc

    def extract(self, signal):
        frames = (len(signal) - self.FRAME_LEN) / self.FRAME_SHIFT + 1
        feature = []
        for f in xrange(frames):
            frame = signal[f * self.FRAME_SHIFT : f * self.FRAME_SHIFT + self.FRAME_LEN]
            feature.append(self.lpcc(frame))
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
