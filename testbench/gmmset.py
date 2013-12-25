#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: gmmset.py
# $Date: Wed Dec 25 01:10:42 2013 +0000
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import operator
import multiprocessing

from collections import defaultdict

import numpy as np

from gmm.python.pygmm import GMM

class GMMSet(object):
    def __init__(self, gmm_order=32, ubm=None,
            **kwargs):
        self.kwargs = kwargs
        self.gmms = []
        self.ubm = ubm
        if ubm is not None:
            self.gmm_order = ubm.get_nr_mixtures()
        else:
            self.gmm_order = gmm_order
        self.y = []

    def fit_new(self, x, label):
        self.y.append(label)
        gmm = GMM(self.gmm_order, **self.kwargs)
        gmm.fit(x, self.ubm)
        self.gmms.append(gmm)

    def cluster_by_label(self, X, y):
        Xtmp = defaultdict(list)
        for ind, x in enumerate(X):
            label = y[ind]
            Xtmp[label].extend(x)
        yp, Xp = zip(*Xtmp.iteritems())
        return Xp, yp

    def fit(self, X, y):
        X, y = self.cluster_by_label(X, y)
        for ind, x in enumerate(X):
            self.fit_new(x, y[ind])

    def gmm_score(self, gmm, x):
        return np.exp(np.sum(gmm.score(x)) / 1000)

    def predict_one(self, x):
        scores = [self.gmm_score(gmm, x) for gmm in self.gmms]
        return self.y[max(enumerate(scores), key=operator.itemgetter(1))[0]]

    def predict(self, X):
        return map(self.predict_one, X)

    def load_gmm(self, label, fname):
        self.y.append(label)
        gmm = GMM.load(fname)
        for key, val in self.kwargs.iteritems():
            exec("gmm.{0} = val".format(key))
        self.gmms.append(gmm)


class GMMSetPyGMM(GMMSet):
    def predict_one(self, x):
        scores = [gmm.score_all(x) for gmm in self.gmms]
        return self.y[max(enumerate(scores), key=operator.itemgetter(1))[0]]

# vim: foldmethod=marker
