#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: plot-wav.py
# $Date: Mon Dec 30 22:03:40 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import numpy as np

def main():
    if not(len(sys.argv) == 2 or len(sys.argv) == 4):
        print "Usage: {} <wav_file> [<start_sec> <end_sec>]" .format(sys.argv[0])
        sys.exit(1)

    fs, signal = wavfile.read(sys.argv[1])
    start, end = 0, 10**9
    st, et = 0, len(signal) / float(fs)
    if len(sys.argv) == 4:
        st, et = map(float, sys.argv[2:4])
        start, end = map(lambda x: int(x * fs), [st, et])

    y = signal[start:end]
    duration = len(y) / float(fs)
    x = np.linspace(st, et, len(y))
    plt.figure()
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    main()


# vim: foldmethod=marker
