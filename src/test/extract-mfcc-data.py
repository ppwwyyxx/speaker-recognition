#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: extract-mfcc-data.py
# $Date: Tue Dec 24 20:23:39 2013 +0000
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>


from sample import Sample
from scipy.io import wavfile
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import glob
import BOB as MFCC
import multiprocessing
import numpy as np
import operator
import errno

concurrency = 8

def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def get_mfcc_worker(params):
    fpath, outpath = params
    print('mfcc: ' + fpath)
    fs, signal = wavfile.read(fpath)
    mfcc = MFCC.extract(fs, signal)
    mkdirp(os.path.dirname(outpath))
    with open(outpath, 'w') as fout:
        for x in mfcc:
            print >> fout, " " . join(map(str, x))

def extract_mfcc_data(dirname, mfcc_output_dir):
    pool = multiprocessing.Pool(concurrency)
    files = glob.glob(os.path.join(dirname, '*.wav'))
    mfcc_files = map(lambda x: os.path.join(mfcc_output_dir, x[0] + ".mfcc"),
        map(os.path.splitext,
            map(os.path.basename, files)))
    result = pool.map(get_mfcc_worker, zip(files, mfcc_files))
    pool.terminate()

def main():
    for style in ['Style_Spontaneous', 'Style_Whisper', 'Style_Reading']:
        dirname = '../test-data/corpus.silence-removed/' + style
        mfcc_output_dir = "mfcc-data/" + style
        extract_mfcc_data(dirname, mfcc_output_dir)

if __name__ == '__main__':
    main()


# vim: foldmethod=marker

