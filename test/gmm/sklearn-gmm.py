#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: gmm.py
# $Date: Tue Dec 10 11:36:41 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

from sklearn.mixture import GMM


def read_data(fname):
    with open(fname) as fin:
        return map(lambda line: map(float, line.rstrip().split()), fin)

def dump_gmm(gmm):
    print gmm.n_components
    print " " . join(map(str, gmm.weights_))
    for i in range(gmm.n_components):
        print len(gmm.means_[i]), 1
        print " " . join(map(str, gmm.means_[i]))
        print " " . join(map(str, gmm.covars_[i]))

def main():
    gmm = GMM(3)
    X = read_data('test.data')
    gmm.fit(X)
    dump_gmm(gmm)

if __name__ == '__main__':
    main()


# vim: foldmethod=marker
