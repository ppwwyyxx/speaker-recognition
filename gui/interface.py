#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: interface.py
# Date: Sat Dec 28 21:39:40 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from collections import defaultdict
from scipy.io import wavfile
import time
import cPickle as pickle
from filters.VAD import VAD

from feature import mix_feature

#from gmmset import GMMSetPyGMM as GMMSet
#from gmmset import GMM
from skgmm import GMMSet, GMM

class ModelInterface(object):

    UBM_MODEL_FILE = 'model/ubm.mixture-32.utt-300.model'

    def __init__(self):
        self.features = defaultdict(list)
        self.gmmset = GMMSet()
        self.vad = VAD()

    def init_noise(self, fs, signal):
        self.vad.init_noise(fs, signal)

    def filter(self, fs, signal):
        ret = self.vad.filter(fs, signal)
        wavfile.write("filtered.wav", fs, ret)
        return ret

    def enroll(self, name, fs, signal):
        feat = mix_feature((fs, signal))
        self.features[name].extend(feat)

    def _get_gmm_set(self):
        from gmmset import GMMSetPyGMM
        if GMMSet is GMMSetPyGMM:
            return GMMSet(ubm=GMM.load(self.UBM_MODEL_FILE))
        else:
            return GMMSet()

    def train(self):
        self.gmmset = self._get_gmm_set()
        start = time.time()
        print "Start training..."
        for name, feats in self.features.iteritems():
            self.gmmset.fit_new(feats, name)
        print time.time() - start, " seconds"

    def predict(self, fs, signal, reject=False):
        from gmmset import GMMSetPyGMM
        if GMMSet is not GMMSetPyGMM:
            reject = False
        print "Lenth:", len(signal)
        try:
            feat = mix_feature((fs, signal))
        except Exception as e:
            print str(e)
            return None
        if reject:
            try:
                l = self.gmmset.predict_one_with_rejection(feat)
                return l
            except Exception as e:
                print str(e)
        return self.gmmset.predict_one(feat)

    def dump(self, fname):
        self.gmmset.before_pickle()
        with open(fname, 'w') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.gmmset.after_pickle()

    @staticmethod
    def load(fname):
        with open(fname, 'r') as f:
            R = pickle.load(f)
            R.gmmset.after_pickle()
            return R



if __name__ == "__main__":
    m = ModelInterface()
    fs, signal = wavfile.read("../corpus.silence-removed/Style_Reading/f_001_03.wav")
    m.enroll('h', fs, signal[:80000])
    fs, signal = wavfile.read("../corpus.silence-removed/Style_Reading/f_003_03.wav")
    m.enroll('a', fs, signal[:80000])
    m.train()
    print m.predict(fs, signal[:80000])
