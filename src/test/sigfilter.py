#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: sigfilter.py
# $Date: Sat Nov 30 17:57:47 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

"""
Filter signals
"""

import numpy as np
from sample import Sample

def sample_adapter(func):
    def sample_adapter_wrapper(fs = None, signal = None):
        if fs.__class__ == Sample:
            return Sample(*func(fs.fs, fs.signal))
        else:
            return Sample(*func(fs, signal))
    return sample_adapter_wrapper

def get_threshold_percentage_filter(perc):
    @sample_adapter
    def signal_filter(fs, signal):
        sig_abs = np.absolute(signal)
        sig_max = np.max(sig_abs)
        sig_threshold = sig_max * perc
        return fs,sig_abs[sig_abs > sig_threshold]

    return signal_filter

def get_speaking_filter(frame_duration = 0.02, frame_shift = 0.01, perc = 0.05):
    @sample_adapter
    def signal_filter(fs, signal):
        siglen = len(signal)
        retsig = np.zeros(siglen)
        frame_length = frame_duration * fs
        frame_shift_length = frame_shift * fs
        new_siglen = 0
        i = 0
        average_energy = np.sum(signal**2) / float(siglen)
        while i < siglen:
            subsig = signal[i:i+frame_length]
            ave_energy = np.sum(subsig**2) / float(len(subsig))
            if ave_energy < average_energy * perc:
                i += frame_length
            else:
                sigaddlen = min(frame_shift_length, len(subsig))
                retsig[new_siglen:new_siglen+sigaddlen] = subsig[:sigaddlen]
                new_siglen += sigaddlen
                i += frame_shift_length
        return fs, retsig[:new_siglen]

    return signal_filter



# vim: foldmethod=marker

