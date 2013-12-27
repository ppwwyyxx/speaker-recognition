#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: utils.py
# Date: Thu Dec 26 18:17:17 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from scipy.io import wavfile

def write_wav(fname, fs, signal):
    wavfile.write(fname, fs, signal)

def time_str(seconds):
    minutes = int(seconds / 60)
    sec = int(seconds % 60)
    return "{:02d}:{:02d}".format(minutes, sec)

if __name__ == "__main__":
    print time_str(100.0)
