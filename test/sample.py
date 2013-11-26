#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: sample.py
# $Date: Tue Nov 26 11:31:11 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import scipy.io.wavfile as wavfile
import numpy as np
import random

class Sample(object):
    def __init__(self, fs = None, signal = None):
        if signal == None:
            signal = np.array([])
        self.signal = signal
        self.fs = fs

    @staticmethod
    def from_wavfile(fname):
        return Sample(*wavfile.read(fname))

    def write(self, fname):
        wavfile.write(fname, self.fs, self.signal)

    def duration(self):
        return len(self.signal) / float(self.fs)

    def add(self, sample):
        if self.fs:
            assert sample.fs == self.fs, "{} != {}" . format(sample.fs, self.fs)
        else:
            self.fs = sample.fs
        self.signal = np.concatenate((self.signal, sample.signal))

    def get_fragment(self, duration):
        tot_duration = self.duration()
        count = min(len(self.signal), int(duration * self.fs))
        pos = random.randint(0, len(self.signal) - count)
        return self.fs, self.signal[pos:pos + count]



# vim: foldmethod=marker

