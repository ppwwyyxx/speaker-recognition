#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: utils.py
# Date: Thu Dec 26 16:41:18 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from scipy.io import wavfile

def write_wav(fname, fs, signal):
    wavfile.write(fname, fs, signal)

def read_wav(fname):
    return wavfile.read(fname)
