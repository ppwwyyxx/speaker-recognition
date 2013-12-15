#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: BOB.py
# Date: Sun Dec 15 19:43:51 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import bob
import numpy

win_length_ms = 20 # The window length of the cepstral analysis in milliseconds
win_shift_ms = 10 # The window shift of the cepstral analysis in milliseconds
#n_filters = 40 # The number of filter bands
#n_ceps = 19 # The number of cepstral coefficients
f_min = 0. # The minimal frequency of the filter bank
#f_max = 6000. # The maximal frequency of the filter bank
delta_win = 2 # The integer delta value used for computing the first and second order derivatives
pre_emphasis_coef = 0.97 # The coefficient used for the pre-emphasis
dct_norm = True # A factor by which the cepstral coefficients are multiplied
mel_scale = True # Tell whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale


mfcc_extractors = dict()
def get_mfcc_extractor(fs, n_ceps=19, f_max=6000, n_filters=40):
    global mfcc_extractors
    if fs not in mfcc_extractors:
        mfcc_extractors[fs] = bob.ap.Ceps(fs, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min,
                f_max, delta_win, pre_emphasis_coef, True, dct_norm)
    return mfcc_extractors[fs]

def extract(fs, signal=None, **kwargs):
    """accept two argument, or one as a tuple"""
    if signal is None:
        assert type(fs) == tuple
        fs, signal = fs[0], fs[1]

    signal = numpy.cast['float'](signal)
    return get_mfcc_extractor(fs, **kwargs)(signal)


if __name__ == "__main__":
    import scipy.io.wavfile
    fs, signal = scipy.io.wavfile.read("../corpus.silence-removed/Style_Reading/f_001_03.wav")
    mfcc = extract(fs, signal)
    print mfcc

