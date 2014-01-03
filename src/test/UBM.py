#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: UBM.py
# Date: Sun Dec 08 00:47:00 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import numpy
import pypr.clustering.gmm as pypr_GMM

class UBM(object):
    def __init__(self, n_component, n_iter):
        self.n_comp = n_component
        self.n_iter = n_iter
        self.samples = []

    def add_sample(self, sample):
        self.samples.append(sample)

    def fit(self):
        l = len(self.samples)
        print "Data Size: ", l
        self.samples = numpy.asarray(self.samples)
        print "start fitting..."
        self.means, self.covars, \
                self.weights, self.logl = pypr_GMM.em_gm(self.samples, K=self.n_comp, max_iter=self.n_iter,
                       verbose=True, diag_add=1e-9)
        #for m in range(self.n_comp):
        #    for d in range(13):
        #        print self.covars[m][d][d]
#       nmixture x 13, nmixture x 13, nmixture

    def dump_paras(self, dirname):
        with open(dirname + "/ubm_means", 'w') as f:
            for vec13 in self.means:
                f.write(' '.join(map(str, vec13)))
                f.write(' ')
            f.write('\n')
        with open(dirname + "/ubm_variances", 'w') as f:
            for mat13 in self.covars:
                for d in xrange(13):
                    f.write(str(mat13[d][d]))
                    f.write(' ')
            f.write('\n')
        with open(dirname + "/ubm_weights", 'w') as f:
            f.write(' '.join(map(str, self.weights)))
            f.write('\n')

    @staticmethod
    def train_from_file(train_lst, n_component, n_iter):
        ubm = UBM(n_component, n_iter)
        with open(train_lst) as f:
            for line in f:
                fname = line.strip().split('=')[1]
                with open(fname) as f_feat:
                    for line in f_feat:
                        vec = numpy.array(map(float, line.strip().split(' ')))
                        ubm.add_sample(vec)
        ubm.fit()
        return ubm

if __name__ == "__main__":
    ubm = UBM.train_from_file('feature-data/enroll.lst', 256, 100)
    dirname = 'models'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    ubm.dump_paras(dirname)
