#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: remove-silence.py
# $Date: Sat Nov 30 18:14:33 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import sys
import os
import glob
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt

def mkdirp(dirname):
    try:
        os.makedirs(dirname)
    except OSError as err:
        if err.errno!=17:
            raise

def remove_silence(fs, signal,
        frame_duration = 0.02,
        frame_shift = 0.01,
        perc = 0.05):
    siglen = len(signal)
    retsig = np.zeros(siglen, dtype = type(signal[0]))
    frame_length = frame_duration * fs
    frame_shift_length = frame_shift * fs
    new_siglen = 0
    i = 0
    # NOTE: signal ** 2 where signal is a numpy array
    #       interpret an unsigned integer as signed integer,
    #       e.g, if dtype is uint8, then
    #           [128, 127, 129] ** 2 = [0, 1, 1]
    #       so the energy of the signal is somewhat
    #       right
    average_energy = np.sum(signal ** 2) / float(siglen)
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
    return fs, retsig[:new_siglen]

def main():
    if len(sys.argv) != 3:
        print("Usage: {} <orignal_dir> <output_dir>" . format(sys.argv[0]))
        sys.exit(1)

    ORIG_DIR, OUTPUT_DIR = sys.argv[1:]
    for style in glob.glob(os.path.join(ORIG_DIR, '*')):
        dirname = os.path.basename(style)
        for fpath in glob.glob(os.path.join(style, '*.wav')):
            fname = os.path.basename(fpath)
            new_fpath = os.path.join(OUTPUT_DIR, dirname, fname)
            mkdirp(os.path.dirname(new_fpath))

            print(fpath)


            fs, signal = wavfile.read(fpath)
            fs_out, signal_out = remove_silence(fs, signal)
            wavfile.write(new_fpath, fs_out, signal_out)

            continue
            f, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(signal)
            ax2.plot(signal_out)
            plt.show()

if __name__ == '__main__':
    main()

# vim: foldmethod=marker

