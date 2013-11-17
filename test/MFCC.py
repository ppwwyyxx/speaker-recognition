#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: MFCC.py
# Date: Sun Nov 17 22:08:41 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

POWER_SPECTRUM_FLOOR = 1e-100

from numpy import *
import numpy.linalg as linalg

class MFCCExtractor(object):

    def __init__(self, fs, FFT_SIZE=2048, n_bands=40, n_COEFS=13,
                 PRE_EMPH=0.95):
        self.PRE_EMPH = PRE_EMPH
        self.fs = fs
        self.bands = n_bands
        self.COEFS = n_COEFS
        self.FFT_SIZE = FFT_SIZE

        self.FRAME_LEN = int(0.02 * fs)
        self.FRAME_SHIFT = int(0.01 * fs)

        self.window = MFCCExtractor.hamming(self.FRAME_LEN)


        self.M, self.CF = MFCCExtractor.melfb(self.bands, self.FFT_SIZE,
                                              self.fs)

        self.D = MFCCExtractor.dctmtx(self.bands)[1: self.COEFS + 1]
        self.invD = linalg.inv(MFCCExtractor.dctmtx(self.bands))[:, 1: self.COEFS + 1]  # The inverse DCT matrix. Change the index to [0:COEFS] if you want to keep the 0-th coefficient


    def extract(self, signal):
        """
        Extract MFCC coefficients of the sound x in numpy array format.
        """

        if signal.ndim > 1:
            print "INFO: Input signal has more than 1 channel; the channels will be averaged."
            signal = mean(signal, axis=1)
        frames = (len(signal) - self.FRAME_LEN) / self.FRAME_SHIFT + 1
        feature = []
        for f in range(frames):
            # Windowing
            frame = signal[f * self.FRAME_SHIFT : f * self.FRAME_SHIFT +
                           self.FRAME_LEN] * self.window
            # Pre-emphasis
            frame[1:] -= frame[:-1] * self.PRE_EMPH
            # Power spectrum
            X = abs(fft.fft(frame, self.FFT_SIZE)[:self.FFT_SIZE / 2 + 1]) ** 2
            X[X < POWER_SPECTRUM_FLOOR] = POWER_SPECTRUM_FLOOR  # Avoid zero
            # Mel filtering, logarithm, DCT
            X = dot(self.D, log(dot(self.M, X)))
            feature.append(X)
        feature = row_stack(feature)
        # Show the MFCC spectrum before normalization
        # Mean & variance normalization
        if feature.shape[0] > 1:
            mu = mean(feature, axis=0)
            sigma = std(feature, axis=0)
            feature = (feature - mu) / sigma
        return feature

    def extract_differential(self, signal):
        feature = self.extract(signal)
        ret = []
        for feat in feature:
            diff = lambda f: [x - f[i - 1] for i, x in enumerate(f)][1:]
            diff1 = diff(feat)
            diff2 = diff(diff1)
            ret.append(concatenate((feat, diff1, diff2)))
        return ret

    @staticmethod
    def differentiate(feature):
        return

    @staticmethod
    def hamming(n):
        """
        Generate a hamming window of n points as a numpy array.
        """
        return 0.54 - 0.46 * cos(2 * pi / n * (arange(n) + 0.5))

    @staticmethod
    def melfb(p, n, fs):
        """
        Return a Mel filterbank matrix as a numpy array.
        Inputs:
            p:  number of filters in the filterbank
            n:  length of fft
            fs: sample rate in Hz
        Ref. http://www.ifp.illinois.edu/~minhdo/teaching/speaker_recognition/code/melfb.m
        """
        f0 = 700.0 / fs
        fn2 = int(floor(n/2))
        lr = log(1 + 0.5/f0) / (p+1)
        CF = fs * f0 * (exp(arange(1, p+1) * lr) - 1)
        bl = n * f0 * (exp(array([0, 1, p, p+1]) * lr) - 1)
        b1 = int(floor(bl[0])) + 1
        b2 = int(ceil(bl[1]))
        b3 = int(floor(bl[2]))
        b4 = min(fn2, int(ceil(bl[3]))) - 1
        pf = log(1 + arange(b1,b4+1) / f0 / n) / lr
        fp = floor(pf)
        pm = pf - fp
        M = zeros((p, 1+fn2))
        for c in range(b2-1,b4):
            r = fp[c] - 1
            M[r,c+1] += 2 * (1 - pm[c])
        for c in range(b3):
            r = fp[c]
            M[r,c+1] += 2 * pm[c]
        return M, CF

    @staticmethod
    def dctmtx(n):
        """
        Return the DCT-II matrix of order n as a numpy array.
        """
        x, y = meshgrid(range(n), range(n))
        D = sqrt(2.0/n) * cos(pi * (2*x+1) * y / (2*n))
        D[0] /= sqrt(2)
        return D
