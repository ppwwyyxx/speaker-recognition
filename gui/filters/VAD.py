#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: VAD.py
# Date: Fri Dec 27 02:50:04 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from noisered import NoiseReduction
from silence import remove_silence

class VAD(object):

    def __init__(self):
        self.initted = False
        self.nr = NoiseReduction()

    def init_noise(self, fs, signal):
        self.initted = True
        self.nr.init_noise(fs, signal)

    def filter(self, fs, signal):
        nred = self.nr.filter(fs, signal)
        removed = remove_silence(fs, nred)
        return removed



if __name__ == "__main__":
    from scipy.io import wavfile
    import sys
    fs, bg = wavfile.read(sys.argv[1])
    vad = VAD()
    vad.init_noise(fs, bg)

    fs, sig = wavfile.read(sys.argv[2])
    vaded = vad.filter(fs, sig)
    wavfile.write('vaded.wav', fs, vaded)

