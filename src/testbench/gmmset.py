#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: gmmset.py
# $Date: Sun Feb 22 20:17:14 2015 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import operator
import multiprocessing

from collections import defaultdict

import numpy as np

from gmm.python.pygmm import GMM

class GMMSet(object):
    def __init__(self, gmm_order=32, ubm=None,
            reject_threshold=10,
            **kwargs):
        self.kwargs = kwargs
        self.gmms = []
        self.ubm = ubm
        self.reject_threshold = reject_threshold
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

    def auto_tune_parameter(self, X, y):
        if self.ubm is None:
            return
        # TODO

    def fit(self, X, y):
        X, y = self.cluster_by_label(X, y)
        for ind, x in enumerate(X):
            self.fit_new(x, y[ind])

        self.auto_tune_parameter(X, y)

    def gmm_score(self, gmm, x):
        return np.sum(gmm.score(x))

    def predict_one_scores(self, x):
        return [self.gmm_score(gmm, x) for gmm in self.gmms]

    def predict_one(self, x):
        scores = self.predict_one_scores(x)
        return self.y[max(enumerate(scores), key=operator.itemgetter(1))[0]]

    def predict(self, X):
        return map(self.predict_one, X)

    def predict_one_with_rejection(self, x):
        assert self.ubm is not None, \
            "UBM must be given prior to conduct reject prediction."
        scores = self.predict_one_scores(x)
        x_len = len(x) # normalize score
        scores = map(lambda v: v / x_len, scores)
        max_tup = max(enumerate(scores), key=operator.itemgetter(1))
        ubm_score = self.gmm_score(self.ubm, x) / x_len
        #print scores, ubm_score
        if max_tup[1] - ubm_score < self.reject_threshold:
            #print max_tup[1], ubm_score, max_tup[1] - ubm_score
            return None
        return self.y[max_tup[0]]

    def predict_with_reject(self, X):
        return map(self.predict_one_with_rejection, X)

    def load_gmm(self, label, fname):
        self.y.append(label)
        gmm = GMM.load(fname)
        for key, val in self.kwargs.iteritems():
            exec("gmm.{0} = val".format(key))
        self.gmms.append(gmm)


class GMMSetPyGMM(GMMSet):
    def predict_one(self, x):
        scores = [gmm.score_all(x) / len(x) for gmm in self.gmms]
        p = sorted(scores)
        #print scores, p[-1] - p[-2]
        return self.y[max(enumerate(scores), key=operator.itemgetter(1))[0]]

    def before_pickle(self):
        self.gmms = [x.dumps() for x in self.gmms]

    def after_pickle(self):
        self.gmms = [GMM.loads(x) for x in self.gmms]



# vim: foldmethod=marker
