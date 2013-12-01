#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: BOB.py
# Date: Sun Dec 01 21:53:47 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import bob
import numpy

win_length_ms = 20 # The window length of the cepstral analysis in milliseconds
win_shift_ms = 10 # The window shift of the cepstral analysis in milliseconds
n_filters = 24 # The number of filter bands
n_ceps = 19 # The number of cepstral coefficients
f_min = 0. # The minimal frequency of the filter bank
f_max = 4000. # The maximal frequency of the filter bank
delta_win = 2 # The integer delta value used for computing the first and second order derivatives
pre_emphasis_coef = 0.97 # The coefficient used for the pre-emphasis
dct_norm = True # A factor by which the cepstral coefficients are multiplied
mel_scale = True # Tell whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale


mfcc_extractors = dict()
def get_mfcc_extractor(fs, verbose = False):
    global mfcc_extractors
    if fs not in mfcc_extractors:
        mfcc_extractors[fs] = bob.ap.Ceps(fs, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min,
                f_max, delta_win, pre_emphasis_coef, True, dct_norm)
    return mfcc_extractors[fs]

#lfcc_extractors = dict()
#def get_lfcc_extractor(fs, verbose = False):
    #global lfcc_extractors
    #if fs not in lfcc_extractors:
        #lfcc_extractors[fs] = bob.ap.Ceps(fs, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min,
                #f_max, delta_win, pre_emphasis_coef, False, dct_norm)
    #return lfcc_extractors[fs]

def extract(fs, signal):
    signal = numpy.cast['float'](signal)
    return get_mfcc_extractor(fs)(signal)


if __name__ == "__main__":
    import scipy.io.wavfile
    fs, signal = scipy.io.wavfile.read("../corpus.silence-removed/Style_Reading/f_001_03.wav")
    mfcc = extract(fs, signal)
    print mfcc

