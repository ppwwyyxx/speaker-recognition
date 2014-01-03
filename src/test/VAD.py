#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: VAD.py
# $Date: Thu Dec 26 13:39:12 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

# Voice Activity Detectionreturn fs, new_signal

import numpy as np
import sys
from numpy import *
from collections import deque
import matplotlib.pyplot as plt

POWER_SPECTRUM_FLOOR = 1e-100

class LTSV(object):
    """
    Long-Term Signal Variability
    """
    def __init__(self, frame_duration = 0.02, frame_time_shift = 0.01,
            nr_dft = 2048, verbose = False):
        self.frame_duration = frame_duration
        self.frame_time_shift = frame_time_shift;
        self.nr_dft = nr_dft
        self.verbose = verbose

    def compute(self, fs, signal):
        """
        return LTSV and its corresponding signal
        """

        K = self.nr_dft * (4000 - 500) / fs
        k_start = self.nr_dft * 500 / fs
        k_end = self.nr_dft * 4000 / fs
        omega = arange(K) / float(K) * (4000 - 500) + 500

        print K, k_start, k_end

        if signal.ndim > 1:
#            self.dprint("LTSV: Input signal has more than 1 channel; the channels will be averaged.")
#            signal = mean(signal, axis=1)
            signal = array(zip(*signal)[0])
            print signal

        frame_length = self.frame_duration * fs
        frame_shift = self.frame_time_shift * fs
        nr_frames = int((len(signal) - frame_length) / frame_shift + 1)
        window = hanning(frame_length)

        # compute L
        L = []
        frame_interval = []
        for fid in range(nr_frames):
            frame_start_pos = int(fid * frame_shift)
            frame_end_pos = int(fid * frame_shift + frame_length)
            frame = signal[frame_start_pos:frame_end_pos]
            frame_interval.append((frame_start_pos, frame_end_pos))
            frame = frame * window

            S = abs(fft.fft(frame)[k_start : k_end]) ** 2
            S[S < POWER_SPECTRUM_FLOOR] = POWER_SPECTRUM_FLOOR # avoid zero
            Z = sum(S)
            s = S / Z
            x = -s * log(s)
            L.append(var(x))
            if False and mean(abs(S)) > 0 and frame_start_pos / float(fs) > 3:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
                ax1.plot((arange(len(frame)) + frame_start_pos) / float(fs),
                        frame / window)
                ax1.set_title('Signal {:.2f}s ~ {:.2f}s' . format(
                    frame_start_pos / float(fs), frame_end_pos / float(fs)))
                nr_pt_plot = 10
                print s[:nr_pt_plot]
                print x[:nr_pt_plot]
                print (x / sum(x))[:nr_pt_plot]
                print log(s[:nr_pt_plot])
                print log(s[:nr_pt_plot]) * s[:nr_pt_plot]
                ax2.plot(omega[:nr_pt_plot], s[:nr_pt_plot])
                ax2.set_title('Spectrum')
                ax3.plot(omega[:nr_pt_plot], x[:nr_pt_plot])
                ax3.set_title('Entropy')
                plt.show()
#                from IPython import embed
#                embed()

        L = array(L)
        return L, frame_interval

    def dprint(self, msg):
        """ Debug print """
        if self.verbose:
            print(msg)

class MonoQueue(object):
    def __init__(self, size, greater = lambda a, b: a > b):
        self.queue = deque()
        self.size = size
        self.greater = greater

    def append(self, ind, val = None):
        if val == None:
            ind, val = ind
        if self.queue:
            assert self.queue[-1][0] < ind
        while len(self.queue) > 0 and ind - self.queue[0][0] >= self.size:
            self.queue.popleft()
        while len(self.queue) > 0 and self.greater(val, self.queue[-1][1]):
            self.queue.pop()
        self.queue.append((ind, val))

    def front_val(self, ind):
        while len(self.queue) > 1 and ind - self.queue[0][0] >= self.size:
            self.queue.popleft()
        if len(self.queue) == 0:
            return 0
        return self.queue[0][1]

    def __getitem__(self, ind):
        return self.queue[ind]

class MonoMaxQueue(MonoQueue):
    def __init__(self, size):
        MonoQueue.__init__(self, size, lambda a, b: a > b)

class MonoMinQueue(MonoQueue):
    def __init__(self, size):
        MonoQueue.__init__(self, size, lambda a, b: a < b)

class VAD(object):
    """
    Voice Activity Dectection
    """
    def __init__(self, **kwargs):
        self.ltsv = LTSV(**kwargs)
        self.threshold_buffer_size = 100
        self.alpha = 0.3
        self.p = 3.0
        self.silence_padding_duration = 1.0
        self.R = 30
        self.c = 0.8
        """
        R: use last R frames for decision, e.g. when frame_shift is 0.01
            and R = 30, then last 0.3s signal are considered
        """

    def vad(self, fs, signal):
        """ return fs, new_signal """
        if signal.ndim > 1:
#            print("INFO: Input signal has more than 1 channel; the channels will be averaged.")
#            signal = mean(signal, axis=1)
            signal = array(zip(*signal)[0])

        print 'signal length: ', len(signal)

        silence_padding_size = int(fs * self.silence_padding_duration)
        silence_padding = [0] * silence_padding_size
        signal_padded = np.concatenate((silence_padding, signal, silence_padding))

        L, frame_interval = self.ltsv.compute(fs, signal_padded)
#        plt.figure()
#        plt.plot(L)
#        plt.show()
#        sys.exit()

        # compute D, decisions for each frame
        B_noise, B_speech = MonoMaxQueue(self.threshold_buffer_size), \
            MonoMinQueue(self.threshold_buffer_size)
        threshold = mean(L[:self.threshold_buffer_size]) + \
                self.p * sqrt(var(L[:self.threshold_buffer_size]))
        D = [0.0] * len(L)
        for i in range(self.threshold_buffer_size):
            if L[i] > threshold:
                B_speech.append(i, L[i])
            else:
                B_noise.append(i, L[i])

        def get_speech_min(D, L, b, e):
            buf = []
            for i in range(b, e):
                if D[i] > 0:
                    buf.append(L[i])
            if len(buf) == 0:
                return 0
            return min(buf)

        def get_noise_max(D, L, b, e):
            buf = []
            for i in range(b, e):
                if D[i] < 1:
                    buf.append(L[i])
            if len(buf) == 0:
                return 0
            return max(buf)

        for i in range(self.threshold_buffer_size, len(L)):
            B_speech_min = get_speech_min(D, L, i-self.threshold_buffer_size, i)
            B_noise_max = get_noise_max(D, L, i-self.threshold_buffer_size, i)


            t0 = self.alpha * B_speech.front_val(i - 1) \
                    + (1.0 - self.alpha) * B_noise.front_val(i - 1)
            threshold = self.alpha * B_speech_min \
                    + (1.0 - self.alpha) * B_noise_max

            assert B_speech.front_val(i - 1) == B_speech_min
            assert B_noise.front_val(i - 1) == B_noise_max
#            print t0, threshold

            if L[i] > threshold:
                D[i] = 1.0
                B_speech.append(i, L[i])
            else:
                D[i] = 0.0
                B_noise.append(i, L[i])

        count = sum(D[0:self.R + 1])

        speech_intervals = []
        for i in range(len(L) - self.R - 1 - 1):
            if count >= self.c * (self.R + 1):
#                print count
                l, r = frame_interval[i]
                if l >= silence_padding_size \
                        and r < len(signal_padded) - silence_padding_size:
                    l -= silence_padding_size
                    r -= silence_padding_size
                    speech_intervals.append((l, r))

            count -= D[i]
            p = i + self.R + 2
            if p < len(L):
                count += D[p]

#        print speech_intervals

        new_signal = []
        cur_left, cur_right = speech_intervals[0]
        for interval in speech_intervals:
            l, r = interval
#            print l, r, cur_left, cur_right
            if l > cur_right:
                print cur_left, cur_right
                new_signal.extend(list(signal[cur_left: cur_right]))
                cur_left, cur_right = l, r
            else:
                cur_right = r
        print cur_left, cur_right
        new_signal.extend(list(signal[cur_left: cur_right]))
        return fs, signal # np.array(new_signal)

# vim: foldmethod=marker

