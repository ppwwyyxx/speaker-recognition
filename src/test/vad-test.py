#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: vad-test.py
# $Date: Sun Nov 24 21:35:13 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

from sample import Sample
from collections import defaultdict
import numpy as np
from numpy import *
import glob
import os
import traceback
import scipy.io.wavfile as wavfile
import VAD
import matplotlib.pyplot as plt

class Person(object):
    def __init__(self, sample = None, name = None, gender = None):
        self.sample = sample
        self.name = name
        self.gender = gender
        self.samples = []

    def add_sample(self, sample):
        self.samples.append(sample)
        if not self.sample:
            self.sample = sample
        else:
            self.sample.add(sample)

    def sample_duration(self):
        return self.sample.duration()

    def get_fragment(self, duration):
        return self.sample.get_fragment(duration)

def get_corpus():
    persons = defaultdict(Person)
    dirs = [
            '../test-data/corpus/Style_Reading',
            '../test-data/corpus/Style_Spontaneous',
            '../test-data/corpus/Style_Whisper',
            ]
    for d in dirs:
        print("processing {} ..." . format(d))
        for fname in glob.glob(os.path.join(d, "*.wav")):
            basename = os.path.basename(fname)
            gender, name, _ = basename.split('_')
            p = persons[name]
            p.name, p.gender = name, gender
            try:
                p.add_sample(Sample.from_wavfile(fname))
            except Exception as e:
                print("Exception occured while reading {}: {} " . format(
                    fname, e))
                print("======= traceback =======")
                print(traceback.format_exc())
                print("=========================")

    return persons

def play_wav(fname):
    import pyaudio
    import wave

    #define stream chunk
    chunk = 1024

    #open a wav format music
    f = wave.open(fname,"rb")
    #instantiate PyAudio
    p = pyaudio.PyAudio()
    #open stream
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                    channels = f.getnchannels(),
                    rate = f.getframerate(),
                    output = True)
    #read data
    data = f.readframes(chunk)

    #paly stream
    while data != '':
        stream.write(data)
        data = f.readframes(chunk)

    #stop stream
    stream.stop_stream()
    stream.close()

    #close PyAudio
    p.terminate()


def add_white_noise(signal):
    import random
    ret = np.copy(signal)
    sd = std(signal)
    for i in range(len(ret)):
        ret[i] += random.gauss(0, sd * 2)
    return ret

def main():
#    nr_person = 1
#    persons = dict(list(get_corpus().iteritems())[:nr_person])
#    person = list(persons.itervalues())[0]
#    fs, signal = person.sample.fs, person.sample.signal

    tmpfname = '/tmp/tmp.wav'
    fname = 'noise-test-2.wav'
    fs, signal = wavfile.read(fname)
#    signal = add_white_noise(signal)
    plt.figure()
    plt.plot(arange(len(signal)) / float(fs), signal)
    plt.title('orignal signal')
    plt.show(block = False)

    vad = VAD.VAD()
    fs, new_signal = vad.vad(fs, signal)

    print 'fs =', fs
    wavfile.write('/tmp/orig.wav', fs, signal)
    wavfile.write('/tmp/x.wav', fs, new_signal)
    plt.figure()
    plt.plot(new_signal)
    plt.show()

#    plt.figure()
#    plt.plot(new_signal)
#    plt.show()

if __name__ == '__main__':
    main()

# vim: foldmethod=marker
