#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: lpcc.py
# Date: Wed Dec 25 13:05:40 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import time
import scikits.talkbox as tb
from scikits.talkbox.linpred import levinson_lpc
import numpy as np
from scipy.io import  wavfile

# lpc order
n_lpc = 10
n_lpcc = 14

def lpc_to_cc(lpc):
    lpcc = np.zeros(n_lpcc)
    lpcc[0] = lpc[0]
    for n in range(1, n_lpc):
        lpcc[n] = lpc[n]
        for l in range(0, n):
            lpcc[n] += lpc[l] * lpcc[n - l - 1] * (n - l) / (n + 1)
    for n in range(n_lpc, n_lpcc):
        lpcc[n] = 0
        for l in range(0, n_lpc):
            lpcc[n] += lpc[l] * lpcc[n - l] * (n - l) / n
    return -lpcc[1:]

def lpcc(signal):
    #lpc = tb.lpc(signal, n_lpc)[0]
    lpc = levinson_lpc.lpc(signal, n_lpc)[0]
    print lpc
    lpcc = lpc_to_cc(lpc)
    return lpcc

def main():
    fs, signal = wavfile.read("./a.wav")

    s = np.array([1, 2, 4, 3, 6, 7, 9, 13, 16, 22, 29, 38])
    out = lpcc(s)
    print out


if __name__ == "__main__":
   main()
