#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: BOB.py
# Date: Wed Oct 29 22:38:35 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from utils import cached_func, diff_feature
import bob
import bob.ap
import numpy

@cached_func
def get_bob_extractor(fs, win_length_ms=32, win_shift_ms=16,
                      n_filters=55, n_ceps=19, f_min=0., f_max=6000,
                      delta_win=2, pre_emphasis_coef=0.95, dct_norm=True,
                      mel_scale=True):
    ret = bob.ap.Ceps(fs, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min,
            f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
    return ret

def extract(fs, signal=None, diff=False, **kwargs):
    """accept two argument, or one as a tuple"""
    if signal is None:
        assert type(fs) == tuple
        fs, signal = fs[0], fs[1]

    signal = numpy.cast['float'](signal)
    ret = get_bob_extractor(fs, **kwargs)(signal)
    if diff:
        return diff_feature(ret)
    return ret
