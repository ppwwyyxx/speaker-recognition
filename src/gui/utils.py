#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: utils.py
# Date: Sat Feb 21 18:43:32 2015 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from scipy.io import wavfile

def read_wav(fname):
    fs, signal = wavfile.read(fname)
    assert len(signal.shape) == 1, "Only Support Mono Wav File!"
    return fs, signal

def write_wav(fname, fs, signal):
    wavfile.write(fname, fs, signal)

def time_str(seconds):
    minutes = int(seconds / 60)
    sec = int(seconds % 60)
    return "{:02d}:{:02d}".format(minutes, sec)

def monophonic(signal):
    if signal.ndim > 1:
        signal = signal[:,0]
    return signal

if __name__ == "__main__":
    print time_str(100.0)
