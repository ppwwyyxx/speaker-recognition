#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: split-data.py
# $Date: Sun Dec 01 20:30:20 2013 +0000
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import sys
import os
import glob
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from sample import Sample

def mkdirp(dirname):
    try:
        os.makedirs(dirname)
    except OSError as err:
        if err.errno!=17:
            raise

def remove_silence(fs, signal,
        frame_duration = 0.02,
        frame_shift = 0.01,
        perc = 0.01):
    orig_dtype = type(signal[0])
    typeinfo = np.iinfo(orig_dtype)
    is_unsigned = typeinfo.min >= 0
    signal = signal.astype(np.int64)
    if is_unsigned:
        signal = signal - typeinfo.max / 2

    siglen = len(signal)
    retsig = np.zeros(siglen, dtype = np.int64)
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
    retsig = retsig[:new_siglen]
    if is_unsigned:
        retsig = retsig + typeinfo.max / 2
    return fs, retsig.astype(orig_dtype)

def split_wav(fpath, outdir):
    print(fpath)
    fs, signal = wavfile.read(fpath)
    train_time = 30.0
    test_time = 5.0
    nr_test = 100

    sample = Sample(fs, signal)
    fs, signal, begin, end = sample.get_fragment_with_interval(30.0)
    sample.remove_subsignal(begin, end)
    wavfile.write(os.path.join(outdir, 'train.wav'), fs, signal)

    for i in range(nr_test):
        fname = os.path.join(outdir, 'test-' + str(i) + '.wav')
        fs, signal = sample.get_fragment(test_time)
        wavfile.write(fname, fs, signal)


def task(fpath, new_fpath):
    fs, signal = wavfile.read(fpath)
    fs_out, signal_out = remove_silence(fs, signal)
    wavfile.write(new_fpath, fs_out, signal_out)
    return fpath

def main():
    if len(sys.argv) != 3:
        print("Usage: {} <orignal_dir> <output_dir>" . format(sys.argv[0]))
        sys.exit(1)

    ORIG_DIR, OUTPUT_DIR = sys.argv[1:]
    pool = multiprocessing.Pool(8)
    result = []
    for style in glob.glob(os.path.join(ORIG_DIR, '*')):
        dirname = os.path.basename(style)
        for fpath in glob.glob(os.path.join(style, '*.wav')):
            fname = os.path.basename(fpath)
            new_fpath = os.path.join(OUTPUT_DIR, dirname, fname)
            basename, extension = os.path.splitext(new_fpath)
            mkdirp(basename)
            result.append(pool.apply_async(split_wav, args = (fpath, basename)))
    pool.close()
    for r in result:
        print(r.get())

if __name__ == '__main__':
    main()

# vim: foldmethod=marker

