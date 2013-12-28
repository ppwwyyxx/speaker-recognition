#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: skgmm.py
# Date: Sat Dec 28 23:12:31 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import operator
import numpy as np
from sklearn.mixture import GMM
class GMMSet(object):

    def __init__(self, gmm_order = 32):
        self.gmms = []
        self.gmm_order = gmm_order
        self.y = []

    def fit_new(self, x, label):
        self.y.append(label)
        gmm = GMM(self.gmm_order)
        gmm.fit(x)
        self.gmms.append(gmm)

    def gmm_score(self, gmm, x):
        return np.sum(gmm.score(x))

    def before_pickle(self):
        pass

    def after_pickle(self):
        pass

    def predict_one(self, x):
        scores = [self.gmm_score(gmm, x) / len(x) for gmm in self.gmms]
        p = sorted(scores)
        print scores, p[-1] - p[-2]
        result = [(self.y[index], value) for (index, value) in enumerate(scores)]
        p = max(result, key=operator.itemgetter(1))
        return p[0]
