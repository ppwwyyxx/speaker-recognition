#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: plot.py
# $Date: Wed Dec 04 13:16:27 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>


from sample import Sample
from scipy.io import wavfile
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import glob
import MFCC
import multiprocessing
import numpy as np

concurrency = 4

def get_mfcc_worker(fpath):
    print('mfcc: ' + fpath)
    fs, signal = wavfile.read(fpath)
    mfcc = MFCC.extract(fs, signal)
    return mfcc[:1500]

def get_mfcc(nr_male = 10, nr_female = 10):
    pool = multiprocessing.Pool(concurrency)
    dirname = '../test-data/corpus.silence-removed/Style_Spontaneous'
    files = glob.glob(os.path.join(dirname, 'm*.wav'))[:nr_male]
    files.extend(glob.glob(os.path.join(dirname, 'f*.wav'))[:nr_female])
    result = pool.map(get_mfcc_worker, files)
    pool.terminate()
    return result[:nr_male], result[nr_male:]

def get_plot_data(mfcc, ind, pieces = 20):
    cnt = defaultdict(int)
    for v in mfcc[:,ind]:
        val = int(v * pieces) / float(pieces)
        cnt[val] += 1
    xs, ys = [], []
    for val, num in sorted(cnt.iteritems()):
        xs.append(val)
        ys.append(num)
    return xs, ys

def plot_diff_speaker(mfcc_male, mfcc_female):
    mfcc_dim = len(mfcc_male[0][0])
    male_color, female_color = 'blue', 'green'

    for ind in range(mfcc_dim):
        print('plotting ' + str(ind))
        plt.figure()
        for color, mfccs in [(male_color, mfcc_male),
                (female_color, mfcc_female)]:
            for mfcc in mfccs:
                xs, ys = get_plot_data(mfcc, ind, pieces)
                plt.plot(xs, ys, color = color)
        plt.savefig('mfcc-diff-speaker{}.png' . format(ind))

def plot_same_speaker(mfcc, fname_prefix, length = 5000):
    mfcc_dim = len(mfcc[0])
    print('mfcc length:' + str(len(mfcc)))
    for ind in range(mfcc_dim):
        print(ind)
        plt.figure()
        for i in range(0, len(mfcc), length):
            xs, ys = get_plot_data(mfcc[i: i + length], ind)
            plt.plot(xs, ys)
        plt.savefig(fname_prefix + str(ind) + '.png')


# 2D plot
def plot2D_same_speaker_worker(x, y, fname):
    print(fname)
    plt.figure()
    plt.scatter(x, y, s = 1, lw = 0)
    plt.savefig(fname)

def plot2D_same_speaker(mfcc, fname_prefix):
    pool = multiprocessing.Pool(concurrency)
    mfcc_dim = len(mfcc[0])
    for i in range(mfcc_dim):
        for j in range(i + 1, mfcc_dim):
            x, y = mfcc[:,i], mfcc[:,j]
            fname = "{}-{}-{}.png" . format(fname_prefix, i, j)
            pool.apply_async(plot2D_same_speaker_worker, args = (x, y, fname))
    pool.close()
    pool.join()
    pool.terminate()

def plot2D_diff_speaker(mfccs, dim0, dim1, fname):
    mfcc_dim = len(mfccs[0][0])
    plt.figure()
    for ind, mfcc in enumerate(mfccs):
        x, y = mfcc[:,dim0], mfcc[:,dim1]
        plt.scatter(x, y, c = [ind] * len(x),
                s = 1, lw = 0)
    plt.savefig(fname)

def main():
    nr_male, nr_female = 2, 2
    pieces = 20
    mfcc_male, mfcc_female = get_mfcc(nr_male, nr_female)
#    plot_diff_speaker(mfcc_male, mfcc_female)
#    plot_same_speaker(mfcc_male[1], 'male-1')
#    plot_same_speaker(mfcc_female[1], 'female-1')
#    plot2D_diff_speaker(mfcc_male, 0, 1, 'male-0-1-2d.png')
    plot2D_same_speaker(mfcc_male[0], 'male-0-2d')
    plot2D_same_speaker(mfcc_male[1], 'male-1-2d')
    plot2D_same_speaker(mfcc_female[0], 'female-0-2d')
    plot2D_same_speaker(mfcc_female[1], 'female-1-2d')


if __name__ == '__main__':
    main()


# vim: foldmethod=marker

