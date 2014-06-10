#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: silence.py
# $Date: Tue Jun 10 15:18:40 2014 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import sys
import scipy.io.wavfile as wavfile
import numpy as np

def remove_silence(fs, signal,
        frame_duration = 0.02,
        frame_shift = 0.01,
        perc = 0.15):
    orig_dtype = type(signal[0])
    typeinfo = np.iinfo(orig_dtype)
    is_unsigned = typeinfo.min >= 0
    signal = signal.astype(np.int64)
    if is_unsigned:
        signal = signal - (typeinfo.max + 1) / 2

    siglen = len(signal)
    retsig = np.zeros(siglen, dtype = np.int64)
    frame_length = int(frame_duration * fs)
    frame_shift_length = int(frame_shift * fs)
    new_siglen = 0
    i = 0
    # NOTE: signal ** 2 where signal is a numpy array
    #       interpret an unsigned integer as signed integer,
    #       e.g, if dtype is uint8, then
    #           [128, 127, 129] ** 2 = [0, 1, 1]
    #       so the energy of the signal is somewhat
    #       right
    average_energy = np.sum(signal ** 2) / float(siglen)
    print average_energy
    #print "Avg Energy: ", average_energy
    while i < siglen:
        subsig = signal[i:i + frame_length]
        ave_energy = np.sum(subsig ** 2) / float(len(subsig))
        if ave_energy < average_energy * perc:
            i += frame_length
        else:
            sigaddlen = min(frame_shift_length, len(subsig))
            retsig[new_siglen:new_siglen + sigaddlen] = subsig[:sigaddlen]
            new_siglen += sigaddlen
            i += frame_shift_length
    retsig = retsig[:new_siglen]
    if is_unsigned:
        retsig = retsig + typeinfo.max / 2
    return retsig.astype(orig_dtype)

def task(fpath, new_fpath):
    fs, signal = wavfile.read(fpath)
    fs_out, signal_out = remove_silence(fs, signal)
    wavfile.write(new_fpath, fs_out, signal_out)
    return fpath

def main():
    task(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()

# vim: foldmethod=marker

