#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gen-features-file.py
# Date: Wed Dec 25 23:53:05 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import glob
import traceback
import sys
import random
import os
import numpy as np
import operator
from collections import defaultdict

from feature import BOB, LPC, MFCC, get_extractor
from sample import Sample
from multiprocess import MultiProcessWorker

class Person(object):
    def __init__(self, sample=None, name=None, gender=None):
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

    def get_fragment_with_interval(self, duration):
        return self.sample.get_fragment_with_interval(duration)

    def remove_subsignal(self, begin, end):
        self.sample.remove_subsignal(begin, end)

def get_corpus(dirs):
    ret = []
    for d in dirs:
        print("processing {} ..." . format(d))
        for fname in sorted(glob.glob(os.path.join(d, "*.wav"))):
            basename = os.path.basename(fname)
            gender, name, _ = basename.split('_')
            p = Person()
            p.name, p.gender = name, gender
            try:
                orig_sample = Sample.from_wavfile(fname)
                p.add_sample(orig_sample)
            except Exception as e:
                print("Exception occured while reading {}: {} " . format(fname, e))
                print("======= traceback =======")
                print(traceback.format_exc())
                print("=========================")
            ret.append(p)

    return ret

class DataSet(object):
    def __init__(self):
        """list item is of format (name, fs, signal)"""
        self.enroll = []
        self.test = []
        self.train = []

    def compute_features(self):
        print("Computing enrollment features...")
        self.enroll_feat = self._compute(self.enroll)
        print("Computing test features...")
        self.test_feat = self._compute(self.test)
        print("Computing train features...")
        self.train_feat = self._compute(self.train)

    def _compute(self, data):
        worker = MultiProcessWorker(BOB.extract)
        args = []
        names = []
        for (name, fs, signal) in data:
            args.append((fs, signal))
            names.append(name)
        rst = worker.run(args)
        del worker
        return zip(names, rst)


    def write(self, dirname):
        print("Writing...")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._write(self.enroll_feat, "enroll", dirname)
        self._write(self.test_feat, "test", dirname)
        self._write(self.train_feat, "train", dirname)

    def _write(self, data, task_name, dirname):
        datalist = []
        for (idx, feat) in enumerate(data):
            person_name = feat[0]
            # feat is an Ax13 matrix
            basename = task_name + "-" + person_name + "-" + str(idx)
            fname = os.path.join(dirname, basename)
            with open(fname, 'w') as f:
                for line in feat[1]:
                    f.write(' '.join(map(str, line)))
                    f.write(' \n')
            datalist.append((person_name, fname))
        with open(os.path.join(dirname, task_name + ".lst"), 'w') as f:
            for line in datalist:
                f.write("{0}={1}\n".format(line[0], line[1]))

if __name__ == "__main__":
    nr_enroll = 30 #
    enroll_duration = 30
    nr_train_fragment = 20
    train_duration = 10
    test_duration = 5
    nr_test_fragment_per_person = 100 #

    persons = get_corpus(['../corpus.silence-removed/Style_Reading'])
    random.shuffle(persons)
    dataset = DataSet()

    print "generating enroll/test data..."
    for p in persons[:nr_enroll]:
        fs, signal, begin, end = p.get_fragment_with_interval(enroll_duration)
        p.remove_subsignal(begin, end)
        dataset.enroll.append((p.name, fs, signal))
        for _ in xrange(nr_test_fragment_per_person):
            fs, signal = p.get_fragment(test_duration)
            dataset.test.append((p.name, fs, signal))

    for p in persons[nr_enroll:]:
        for _ in xrange(nr_train_fragment):
            fs, signal = p.get_fragment(train_duration)
            dataset.train.append((p.name, fs, signal))

    dataset.compute_features()
    dataset.write('feature-data')

