#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: noisered.py
# Date: Fri Dec 27 04:23:28 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import sys
from scipy.io import wavfile
import os
from random import Random
from silence import remove_silence

from utils import monophonic

NOISE_WAV = "/tmp/noise.wav"
NOISE_MODEL = "/tmp/noise.prof"
THRES = 0.21
r = Random()
class NoiseReduction(object):

    def init_noise(self, fs, signal):
        wavfile.write(NOISE_WAV, fs, signal)
        os.system("sox {0} -n noiseprof {1}".format(NOISE_WAV, NOISE_MODEL))

    def filter(self, fs, signal):
        rand = r.randint(1, 100000)
        fname = "/tmp/tmp{0}.wav".format(rand)
        signal = monophonic(signal)
        wavfile.write(fname, fs, signal)
        fname_clean = "/tmp/tmp{0}-clean.wav".format(rand)
        os.system("sox {0} {1} noisered {2} {3}".format(fname, fname_clean,
                                                        NOISE_MODEL, THRES))
        fs, signal = wavfile.read(fname_clean)
        signal = monophonic(signal)

        os.remove(fname)
        os.remove(fname_clean)
        return signal


if __name__ == "__main__":
    fs, bg = wavfile.read(sys.argv[1])
    nr = NoiseReduction()
    nr.init_noise(fs, bg)

    fs, sig = wavfile.read(sys.argv[2])
    vaded = nr.filter(fs, sig)
    wavfile.write('vaded.wav', fs, vaded)

    removed = remove_silence(fs, vaded)
    wavfile.write("removed.wav", fs, removed)
